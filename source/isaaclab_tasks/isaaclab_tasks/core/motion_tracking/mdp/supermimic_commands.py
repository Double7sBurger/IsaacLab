# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-motion references and per-environment physical context for model-based tracking."""

import math

import torch
import warp as wp
from newton import ModelFlags

from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.utils import configclass
from isaaclab.utils.datasets import MotionLoader

from .commands import MotionCommand, MotionCommandCfg
from .kinematics import ArticulationKinematics
from .reset_sampling import ReturnSamplerCfg


class SuperMimicMotionCommand(MotionCommand):
    """Sample a continuous reference within one clip, including during imagined rollouts."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not math.isclose(self.motion.fps * env.step_dt, 1.0):
            raise ValueError("Motion sampling rate must match the policy frequency.")
        clips = [self.motion]
        for path in cfg.additional_motion_files:
            clip = MotionLoader(path, cfg.body_names, self.robot.joint_names, self.device)
            if clip.fps != self.motion.fps:
                raise ValueError("All motion clips must have the same sampling rate.")
            clips.append(clip)
        self.motion_ends = torch.tensor([clip.time_step_total for clip in clips], device=self.device).cumsum(0)
        for name in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            setattr(self.motion, name, torch.cat([getattr(clip, name) for clip in clips]))
        self.motion.time_step_total = sum(clip.time_step_total for clip in clips)
        self.bin_count = int(self.motion.time_step_total * env.step_dt) + 1
        self.sampling_distribution = torch.full((self.bin_count,), 1 / self.bin_count, device=self.device)
        self.bin_failed_count = torch.zeros(self.bin_count, device=self.device)
        self._current_bin_failed = torch.zeros_like(self.bin_failed_count)
        self.encoder_bias = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        self.friction = torch.ones(self.num_envs, 1, device=self.device)
        self.context_body_id = self.robot.body_names.index(cfg.context_body)
        self.nominal_com = self.robot.data.body_com_pos_b.torch[:, self.context_body_id].clone()
        self.com = self.nominal_com.clone()
        self.kinematics = ArticulationKinematics(self.robot, cfg.body_names, self.device)
        self.future_offsets = torch.arange(cfg.goal_frames, device=self.device) * cfg.goal_stride
        foot_ids = self.robot.find_bodies(cfg.friction_bodies)[0]
        shapes = [
            shape
            for body in self.robot.map_body_ids_to_backend(foot_ids)
            for shape in self.robot.root_view.body_shapes[body]
        ]
        self.foot_shapes = torch.tensor(shapes, dtype=torch.long, device=self.device)
        self.com_range = torch.tensor([0.025, 0.05, 0.05], device=self.device)

    def clip_end(self, frames):
        """Return the exclusive end for each global frame index."""
        return self.motion_ends[torch.searchsorted(self.motion_ends, frames.contiguous(), right=True)]

    def window(self, frames):
        """Clamp future targets to the current clip instead of crossing into another motion."""
        return torch.minimum(frames[..., None] + self.future_offsets, self.clip_end(frames)[..., None] - 1)

    def set_sampling_distribution(self, weights):
        """Apply return-based reset weights and expose their concentration in task metrics."""
        self.sampling_distribution.copy_(weights)
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum()
        self.metrics["sampling_entropy"][:] = entropy / max(math.log(self.bin_count), 1e-12)
        probability, index = weights.max(0)
        self.metrics["sampling_top1_prob"][:] = probability
        self.metrics["sampling_top1_bin"][:] = index.float() / self.bin_count

    def _resample_command(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if ids.numel() == 0:
            return
        self.friction[ids] = 1.0
        self.encoder_bias[ids] = 0.0
        self.com[ids] = self.nominal_com[ids]
        if self.cfg.randomize_context:
            self.friction[ids] = torch.empty_like(self.friction[ids]).uniform_(0.3, 1.2)
            self.encoder_bias[ids] = torch.empty_like(self.encoder_bias[ids]).uniform_(-0.01, 0.01)
            self.com[ids] += torch.empty_like(self.com[ids]).uniform_(-1, 1) * self.com_range
        self.robot.set_coms_index(coms=self.com[ids].unsqueeze(1), body_ids=[self.context_body_id], env_ids=ids)
        physics = self._env.sim.backend_manager
        model, view = physics.backend.model, self.robot.root_view
        values = wp.to_torch(view.get_attribute("shape_material_mu", model)).clone()
        values[ids[:, None], 0, self.foot_shapes[None, :]] = self.friction[ids]
        view.set_attribute("shape_material_mu", model, wp.from_torch(values))
        physics.add_model_change(ModelFlags.SHAPE_PROPERTIES)
        super()._resample_command(env_ids)
        # The shared reset writes COM velocity; the motion reference specifies link-origin velocity.
        self.robot.write_root_link_velocity_to_sim_index(
            root_velocity=self.robot.data.root_com_vel_w.torch[ids].clone(), env_ids=ids
        )
        self.robot.actuators.target_command.set_position_index(value=self.robot.data.joint_pos.torch[ids], env_ids=ids)

    def _adaptive_sampling(self, env_ids):
        if self.cfg.sampling_mode == "adaptive":
            super()._adaptive_sampling(env_ids)
            return
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if self.cfg.sampling_mode == "start":
            starts = torch.cat((self.motion_ends.new_zeros(1), self.motion_ends[:-1]))
            frames = starts[ids % len(starts)]
        elif self.cfg.sampling_mode == "uniform":
            frames = torch.randint(self.motion.time_step_total, (len(ids),), device=self.device)
        else:
            bins = torch.multinomial(self.sampling_distribution, len(ids), replacement=True)
            frames = (
                (bins + torch.rand(len(ids), device=self.device)) / self.bin_count * self.motion.time_step_total
            ).long()
            frames.clamp_max_(self.motion.time_step_total - 1)
        self.time_steps[ids] = frames

    def _update_command(self):
        self.time_steps.copy_(
            torch.minimum(self.time_steps + (self._env.episode_length_buf > 0), self.clip_end(self.time_steps) - 1)
        )
        self._update_reference_alignment()
        self.bin_failed_count.lerp_(self._current_bin_failed, self.cfg.adaptive_alpha)
        self._current_bin_failed.zero_()


@configclass
class SuperMimicMotionCommandCfg(MotionCommandCfg):
    """Named motion arrays, future goal window and physical context randomization."""

    class_type: type = SuperMimicMotionCommand
    additional_motion_files: tuple[str, ...] = ()
    goal_frames: int = 10
    goal_stride: int = 5
    context_body: str = "torso_link"
    friction_bodies: tuple[str, ...] = (".*ankle_roll_link",)
    randomize_context: bool = True
    sampling_mode: str = "return"
    return_sampler: ReturnSamplerCfg = ReturnSamplerCfg()


class BiasedJointPositionAction(JointPositionAction):
    """Apply the same encoder offset to policy observations and commanded joint positions."""

    def process_actions(self, actions):
        super().process_actions(actions)
        self._processed_actions.sub_(self._env.command_manager.get_term("motion").encoder_bias)
