# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# Portions adapted from HybridRobotics/SuperMimic under Apache-2.0.
# See docs/licenses/dependencies/supermimic-license.txt.

"""Generalized-coordinate dynamics and differentiable reference-motion objectives."""

import math

import torch
from rsl_rl.algorithms.supermimic_task import DifferentiableTask, State

from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    yaw_quat,
)

from .mdp.reset_sampling import ReturnSampler


class GenCoordStateRepresentation:
    """Predict velocity increments and integrate root pose and joint angles at the policy rate."""

    def __init__(self, num_joints, dt, decimation, device):
        self.widths = (3, 4, num_joints, 3, 3, num_joints)
        self.increment_dim = 6 + num_joints
        self.dt = dt
        self.substep_coefficient = (decimation + 1) / (2 * decimation)
        self.error_floor = torch.full((self.increment_dim,), 0.01, device=device)
        self.gravity = torch.tensor([0.0, 0.0, -1.0], device=device)

    def unflatten(self, flat):
        return flat.split(self.widths, dim=-1)

    def features(self, state):
        p, q, joints, v, w, speeds = state
        return torch.cat(
            (
                p[..., 2:3],
                quat_apply_inverse(q, self.gravity.expand_as(p)),
                quat_apply_inverse(q, v),
                quat_apply_inverse(q, w),
                joints,
                speeds,
            ),
            -1,
        )

    def increments(self, state, next_state):
        return torch.cat(
            (
                quat_apply_inverse(state[1], next_state[3] - state[3]),
                quat_apply_inverse(state[1], next_state[4] - state[4]),
                next_state[5] - state[5],
            ),
            -1,
        )

    def integrate(self, state, increment):
        p, q, joints, v, w, speeds = state
        dv, dw, dj = increment.split((3, 3, self.widths[2]), dim=-1)
        dv, dw = quat_apply(q, dv), quat_apply(q, dw)
        c = self.substep_coefficient
        omega = (w + c * dw) * self.dt
        angle = torch.sqrt(omega.square().sum(-1, keepdim=True) + 1e-8)
        delta = torch.cat((torch.sin(angle / 2) * omega / angle, torch.cos(angle / 2)), -1)
        orientation = torch.nn.functional.normalize(quat_mul(delta, q), dim=-1)
        return (
            p + (v + c * dv) * self.dt,
            orientation,
            joints + (speeds + c * dj) * self.dt,
            v + dv,
            w + dw,
            speeds + dj,
        )

    def error(self, prediction, target):
        return self.increments(target, prediction)


def physical_state(env):
    """Read noiseless, environment-local root pose, joint positions and world-frame velocities."""
    data = env.scene["robot"].data
    pose, velocity = data.root_link_pose_w.torch, data.root_link_vel_w.torch
    return torch.cat(
        (
            pose[:, :3] - env.scene.env_origins,
            pose[:, 3:],
            data.joint_pos.torch,
            velocity[:, :3],
            velocity[:, 3:],
            data.joint_vel.torch,
        ),
        -1,
    )


def reference_phase(env):
    return env.command_manager.get_term("motion").time_steps[:, None].float()


def physical_context(env):
    command = env.command_manager.get_term("motion")
    return torch.cat((command.friction, command.com, command.encoder_bias), -1)


def motion_finished(env):
    command = env.command_manager.get_term("motion")
    return command.time_steps >= command.clip_end(command.time_steps) - 1


def height_failure(env, threshold=0.25):
    command = env.command_manager.get_term("motion")
    actual = env.scene["robot"].data.root_link_pose_w.torch[:, 2] - env.scene.env_origins[:, 2]
    target = command.motion.body_pos_w[command.time_steps, 0, 2]
    return (actual - target).abs() > threshold


def orientation_failure(env, threshold=0.8):
    command = env.command_manager.get_term("motion")
    actual = env.scene["robot"].data.root_link_pose_w.torch[:, 3:]
    target = command.motion.body_quat_w[command.time_steps, 0]
    return (2 * (actual[:, :2].square().sum(-1) - target[:, :2].square().sum(-1))).abs() > threshold


def tracking_reward(env):
    """Evaluate the same objective used by the learner on physical successor states."""
    command = env.command_manager.get_term("motion")
    data, ids = command.robot.data, command.body_indices
    pose, velocity = data.body_link_pose_w.torch[:, ids], data.body_link_vel_w.torch[:, ids]
    frames = command.time_steps
    return body_tracking_reward(
        pose[..., :3] - env.scene.env_origins[:, None],
        matrix_from_quat(pose[..., 3:]),
        velocity[..., :3],
        velocity[..., 3:],
        data.joint_pos.torch,
        env.action_manager.action,
        env.action_manager.prev_action,
        command.motion.body_pos_w[frames],
        command.motion.body_quat_w[frames],
        command.motion.body_lin_vel_w[frames],
        command.motion.body_ang_vel_w[frames],
        data.soft_joint_pos_limits.torch[0],
        pose[:, 0, 3:],
    )


def body_tracking_reward(
    pos, rot, vel, ang, joints, actions, previous, target_pos, target_quat, target_vel, target_ang, limits, root_quat
):
    """Root-aligned Gaussian body tracking, action-rate and soft joint-limit penalties."""
    target_rot = matrix_from_quat(target_quat)
    delta = matrix_from_quat(yaw_quat(quat_mul(root_quat, quat_inv(target_quat[..., 0, :]))))
    origin = torch.cat((pos[..., 0, :2], target_pos[..., 0, 2:3]), -1)
    aligned_pos = origin[..., None, :] + (
        delta[..., None, :, :] @ (target_pos - target_pos[..., :1, :])[..., None]
    ).squeeze(-1)
    aligned_rot = delta[..., None, :, :] @ target_rot

    def rotation_error(a, b):
        cosine = ((a * b).sum((-1, -2)) - 1) * 0.5
        return torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7)).square()

    reward = 0.5 * torch.exp(-(pos[..., 0, :] - target_pos[..., 0, :]).square().sum(-1) / 0.3**2)
    reward += 0.5 * torch.exp(-rotation_error(rot[..., 0, :, :], target_rot[..., 0, :, :]) / 0.4**2)
    reward += torch.exp(-(pos - aligned_pos).square().sum(-1).mean(-1) / 0.3**2)
    reward += torch.exp(-rotation_error(rot, aligned_rot).mean(-1) / 0.4**2)
    reward += torch.exp(-(vel - target_vel).square().sum(-1).mean(-1))
    reward += torch.exp(-(ang - target_ang).square().sum(-1).mean(-1) / 3.14**2)
    joint_limit = ((limits[:, 0] - joints).clamp(min=0) + (joints - limits[:, 1]).clamp(min=0)).sum(-1)
    return reward - 0.1 * (actions - previous).square().sum(-1) - 2.0 * joint_limit


class MotionTrackingTask(DifferentiableTask):
    """Supply reference windows, proprioception and differentiable motion-tracking objectives to RSL-RL."""

    action_history_length = 1

    def __init__(self, env, device="cpu"):
        owner = env.unwrapped
        self.env = owner
        self.command = owner.command_manager.get_term("motion")
        self.motion = self.command.motion
        self.kinematics = self.command.kinematics
        self.dt = owner.step_dt
        self.num_joints = self.command.robot.num_joints
        self.representation = GenCoordStateRepresentation(self.num_joints, self.dt, owner.cfg.decimation, device)
        self.default_pos = self.command.robot.data.default_joint_pos.torch[0].to(device).clone()
        self.limits = self.command.robot.data.soft_joint_pos_limits.torch[0].to(device).clone()
        self.noise_bounds = torch.tensor(
            [0.05] * 3 + [0.2] * 3 + [0.01] * self.num_joints + [0.5] * self.num_joints + [0.0] * self.num_joints,
            device=device,
        )
        action_term = owner.action_manager.get_term("joint_pos")
        self.action_scale = action_term._scale[0].to(device).clone()
        self.action_offset = self.default_pos.clone()
        self.context_mean = torch.tensor([0.75] + [0.0] * (3 + self.num_joints), device=device)
        self.context_scale = torch.tensor([0.45, 0.025, 0.05, 0.05] + [0.01] * self.num_joints, device=device)
        self.enable_noise = owner.cfg.commands.motion.randomize_context
        self.gravity = torch.tensor([0.0, 0.0, -1.0], device=device)
        self.return_sampler = (
            ReturnSampler(
                self.command.cfg.return_sampler,
                self.command.bin_count,
                self.motion.time_step_total,
                owner.num_envs,
                max_step_reward=5.0 * self.dt,
                device=device,
            )
            if self.command.cfg.sampling_mode == "return"
            else None
        )

    def process_env_step(self, goal, rewards, dones, extras):
        if self.return_sampler is None:
            return
        manager = self.env.termination_manager
        truncated = manager.time_outs
        clip_end = manager.get_term("motion_end")
        weights = self.return_sampler.record(
            goal[:, 0].long(), rewards, manager.terminated & ~truncated, clip_end, truncated & ~clip_end
        )
        if weights is not None:
            self.command.set_sampling_distribution(weights)

    def actor_obs(self, state: State, goal, context, prev_actions):
        p, q, joints, v, w, speeds = state
        bias = context[..., 4:]
        return torch.cat(
            (
                quat_apply_inverse(q, self.gravity.expand_as(p)),
                quat_apply_inverse(q, w),
                joints - self.default_pos + bias,
                speeds,
                prev_actions[..., -1, :],
            ),
            -1,
        )

    def actor_goal(self, state, goal):
        indices = self.command.window(goal[..., 0].long())
        content = torch.cat((self.motion.joint_pos[indices], self.motion.joint_vel[indices]), -1).flatten(-2)
        reference = self.motion.body_quat_w[indices, 0]
        relative = quat_mul(quat_inv(state[1])[..., None, :].expand_as(reference), reference)
        rotation = matrix_from_quat(relative)[..., :2].flatten(-3)
        return torch.cat((content, rotation), -1)

    def advance(self, next_state, goal, context):
        frames = goal[..., 0].long()
        return torch.minimum(frames + 1, self.command.clip_end(frames) - 1)[..., None].float(), context

    def dynamics_actions(self, actions):
        return actions * self.action_scale + self.action_offset

    def dynamics_context(self, state, context, prev_actions):
        return (context - self.context_mean) / self.context_scale

    def critic_obs(self, state, goal, context, prev_actions):
        indices = self.command.window(goal[..., 0].long())
        target = self.motion.body_pos_w[indices, 0]
        relative = quat_apply_inverse(
            state[1][..., None, :].expand(*target.shape[:-1], 4), target - state[0][..., None, :]
        )
        actor_goal = self.actor_goal(state, goal)
        content, orientation = actor_goal.split(
            (self.command.cfg.goal_frames * 2 * self.num_joints, self.command.cfg.goal_frames * 6), dim=-1
        )
        return torch.cat(
            (
                self.representation.features(state),
                content,
                relative.flatten(-2),
                orientation,
                self.dynamics_context(state, context, prev_actions),
            ),
            -1,
        )

    def reward(self, state, actions, next_state, goal, prev_actions, labels=None):
        p, q, joints, v, w, speeds = next_state
        positions, rotations, velocities, angular = self.kinematics.forward(p, q, v, w, joints, speeds)
        frames = goal[..., 0].long()
        return self.dt * body_tracking_reward(
            positions,
            rotations,
            velocities,
            angular,
            joints,
            actions,
            prev_actions[..., -1, :],
            self.motion.body_pos_w[frames],
            self.motion.body_quat_w[frames],
            self.motion.body_lin_vel_w[frames],
            self.motion.body_ang_vel_w[frames],
            self.limits,
            q,
        )

    def termination_prob(self, next_state, goal, labels=None):
        frames = goal[..., 0].long()
        height = (next_state[0][..., 2] - self.motion.body_pos_w[frames, 0, 2]).abs() / 0.25
        actual = 2 * next_state[1][..., :2].square().sum(-1)
        target = 2 * self.motion.body_quat_w[frames, 0, :2].square().sum(-1)
        tilt = (actual - target).abs() / 0.8
        return torch.sigmoid((torch.maximum(height, tilt) - 1) / 0.05)

    def world_model_loss(self, prediction, target, scale):
        pos = (prediction[0] - target[0]) / 0.01
        rotation = (matrix_from_quat(prediction[1]) - matrix_from_quat(target[1])).flatten(-2) / (0.02 * math.sqrt(2))
        joints = (prediction[2] - target[2]) / 0.1
        speed = (prediction[5] - target[5]) / 3.14
        return torch.cat((pos, rotation, joints, speed), -1).square().mean(-1)

    def add_noise(self, observation):
        if self.enable_noise:
            return observation + (torch.rand_like(observation) * 2 - 1) * self.noise_bounds
        return observation
