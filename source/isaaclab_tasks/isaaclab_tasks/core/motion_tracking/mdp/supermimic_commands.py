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

    def _update_command(self):
        ended = self.time_steps >= self.clip_end(self.time_steps) - 1
        self.time_steps += ~ended
        self._resample_command(torch.where(ended)[0])
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


class BiasedJointPositionAction(JointPositionAction):
    """Apply the same encoder offset to policy observations and commanded joint positions."""

    def process_actions(self, actions):
        super().process_actions(actions)
        self._processed_actions.sub_(self._env.command_manager.get_term("motion").encoder_bias)
