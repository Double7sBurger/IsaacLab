# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Startup event terms for applying PACE SysID parameters to Newton simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def apply_pace_newton_params(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    pace_checkpoint: str,
    joint_order: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Apply PACE-identified Newton parameters (armature + Coulomb friction) at startup.

    Loads a PACE mean checkpoint (49-dim tensor) and writes the identified armature
    and static friction values into the Newton simulation. Viscous friction, encoder
    bias, and delay are skipped because they require the PaceDCMotor actuator model
    which is not used in the rough-terrain training environment.

    PACE parameter layout (49 values):
        [0:12]  armature per joint [kg·m²]
        [12:24] viscous friction per joint (Newton unsupported — skipped)
        [24:36] Coulomb friction per joint [N·m]
        [36:48] encoder bias per joint [rad] (requires PaceDCMotor — skipped)
        [48]    action delay in sim steps (requires PaceDCMotor — skipped)

    Args:
        env: The RL environment.
        env_ids: Environment indices to apply to (all envs on startup).
        pace_checkpoint: Absolute path to the PACE mean_xxx.pt checkpoint.
        joint_order: Joint names in the order used during PACE data collection.
        asset_cfg: Scene entity config identifying the robot articulation.
    """
    asset = env.scene[asset_cfg.name]
    device = env.device

    params = torch.load(pace_checkpoint, map_location=device)

    armature = params[:12]
    friction = params[24:36]

    # Map PACE joint order → articulation joint indices (order-safe)
    joint_ids = torch.tensor(
        [asset.joint_names.index(name) for name in joint_order],
        device=device,
        dtype=torch.int32,
    )

    num_envs = env.num_envs
    all_env_ids = torch.arange(num_envs, device=device, dtype=torch.int32)

    armature_expanded = armature.unsqueeze(0).expand(num_envs, -1)
    friction_expanded = friction.unsqueeze(0).expand(num_envs, -1)

    asset.write_joint_armature_to_sim(armature_expanded, joint_ids=joint_ids, env_ids=all_env_ids)

    # Use _index variant directly to avoid the deprecated wrapper's full_data kwarg
    # which is absent in the Newton Articulation implementation.
    asset.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=friction_expanded, joint_ids=joint_ids, env_ids=all_env_ids
    )
