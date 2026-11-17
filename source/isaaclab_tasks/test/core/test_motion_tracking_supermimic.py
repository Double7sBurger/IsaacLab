# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared robot configuration and differentiable motion-tracking contracts."""

import math

import torch

from isaaclab.utils.math import matrix_from_quat

from isaaclab_tasks.core.motion_tracking.config.g1.supermimic_env_cfg import G1SuperMimicEnvCfg
from isaaclab_tasks.core.motion_tracking.config.g1.tracking_env_cfg import G1MotionTrackingEnvCfg
from isaaclab_tasks.core.motion_tracking.mdp.supermimic_commands import SuperMimicMotionCommand
from isaaclab_tasks.core.motion_tracking.supermimic import GenCoordStateRepresentation, MotionTrackingTask


def test_supermimic_reuses_motion_tracking_robot_and_control():
    """Choosing a learner does not replace the asset, gains, joint ordering or control rate."""
    ordinary, model_based = G1MotionTrackingEnvCfg(), G1SuperMimicEnvCfg()
    assert model_based.scene.robot.to_dict() == ordinary.scene.robot.to_dict()
    assert model_based.actions.joint_pos.scale == ordinary.actions.joint_pos.scale
    assert model_based.sim.dt == ordinary.sim.dt
    assert model_based.decimation == ordinary.decimation


def test_gencoord_matches_four_semi_implicit_physics_steps():
    """Constant acceleration integrates position and orientation while retaining action derivatives."""
    representation = GenCoordStateRepresentation(1, dt=0.02, decimation=4, device="cpu")
    state = (
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        torch.zeros(1, 1),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.zeros(1, 3),
        torch.tensor([[2.0]]),
    )
    increment = torch.tensor([[0.4, 0.0, 0.0, 0.0, 0.0, 1.0, 0.8]], requires_grad=True)
    prediction = representation.integrate(state, increment)
    torch.testing.assert_close(prediction[0], torch.tensor([[0.025, 0.0, 1.0]]))
    torch.testing.assert_close(prediction[2], torch.tensor([[0.05]]))
    torch.testing.assert_close(prediction[1], torch.tensor([[0.0, 0.0, math.sin(0.00625), math.cos(0.00625)]]))
    torch.testing.assert_close(representation.increments(state, prediction), increment)
    (prediction[0].sum() + prediction[2].sum() + matrix_from_quat(prediction[1]).sum()).backward()
    assert torch.isfinite(increment.grad).all() and increment.grad.abs().sum() > 0


def test_motion_windows_and_imagination_stop_at_each_clip_end():
    """Mixed-length clips retain their own future targets at both sides of a boundary."""
    command = object.__new__(SuperMimicMotionCommand)
    command.motion_ends = torch.tensor([3, 10])
    command.future_offsets = torch.tensor([0, 2, 4])
    frames = torch.tensor([0, 2, 3, 8, 9])
    torch.testing.assert_close(
        command.window(frames), torch.tensor([[0, 2, 2], [2, 2, 2], [3, 5, 7], [8, 9, 9], [9, 9, 9]])
    )
    task = object.__new__(MotionTrackingTask)
    task.command = command
    goal, context = task.advance((), frames[:, None].float(), torch.zeros(5, 1))
    torch.testing.assert_close(goal[:, 0], torch.tensor([1.0, 2.0, 4.0, 9.0, 9.0]))
    torch.testing.assert_close(context, torch.zeros(5, 1))


def test_gencoord_loss_supervises_integrated_pose_not_only_velocity():
    """An incorrect root location must train the model even when predicted velocities are exact."""
    task = object.__new__(MotionTrackingTask)
    representation = GenCoordStateRepresentation(1, dt=0.02, decimation=4, device="cpu")
    state = representation.unflatten(
        torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    )
    prediction = tuple(value.clone().requires_grad_() for value in state)
    target = tuple(value.clone() for value in state)
    target[0][:, 0] += 0.01
    loss = task.world_model_loss(prediction, target, torch.ones(7))
    torch.testing.assert_close(loss, torch.tensor([1 / 14]))
    loss.sum().backward()
    assert prediction[0].grad[0, 0] < 0
