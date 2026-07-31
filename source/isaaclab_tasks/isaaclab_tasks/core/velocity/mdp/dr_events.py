# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain-randomization event terms for explicit (non-implicit) actuator models.

:func:`isaaclab.envs.mdp.events.randomize_actuator_gains` randomizes ``stiffness``
and ``damping``. Those two quantities only reach the simulation through an
:class:`~isaaclab.actuators.ImplicitActuator`'s PD controller. Robots driven by an
explicit model -- :class:`~isaaclab.actuators.ActuatorNetLSTM` on ANYmal-D, for
instance -- compute torque from a learned network and never read either gain, so
randomizing them changes nothing.

For those robots the actuation path that *is* live is the DC-motor torque-speed
clip in :meth:`~isaaclab.actuators.DCMotor._clip_effort`, which every explicit
ANYdrive-style model inherits. :func:`randomize_dc_motor_limits` randomizes that
envelope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

import torch

from isaaclab.actuators import DCMotor
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _sample(
    default: torch.Tensor,
    params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Sample a perturbation of ``default`` with the shape of ``default``."""
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(f"Unknown distribution: '{distribution}'. Use 'uniform', 'log_uniform', 'gaussian'.")

    samples = dist_fn(*params, tuple(default.shape), device=default.device)
    if operation == "add":
        return default + samples
    if operation == "scale":
        return default * samples
    if operation == "abs":
        return samples
    raise NotImplementedError(f"Unknown operation: '{operation}'. Use 'add', 'scale', or 'abs'.")


def randomize_dc_motor_limits(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    effort_limit_distribution_params: tuple[float, float] | None = None,
    velocity_limit_distribution_params: tuple[float, float] | None = None,
    saturation_effort_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
) -> None:
    """Randomize the torque-speed envelope of every :class:`~isaaclab.actuators.DCMotor` actuator.

    The three randomized quantities are the ones :meth:`~isaaclab.actuators.DCMotor._clip_effort`
    reads to bound the torque the actuator is allowed to apply:

    .. code-block:: text

        max_effort = clip(saturation_effort * (1 - joint_vel / velocity_limit), max=effort_limit)
        min_effort = clip(saturation_effort * (-1 - joint_vel / velocity_limit), min=-effort_limit)

    Because the clip is applied to the torque produced by the actuator model, this works for
    explicit learned models (:class:`~isaaclab.actuators.ActuatorNetLSTM`,
    :class:`~isaaclab.actuators.ActuatorNetMLP`) as well as the analytic ones -- unlike
    stiffness/damping randomization, which those models ignore.

    Actuators that do not derive from :class:`~isaaclab.actuators.DCMotor` are skipped.

    The defaults are captured on the first call, so repeated invocations randomize around the
    original configured values rather than compounding.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize. ``None`` randomizes every environment.
        asset_cfg: The articulation whose actuators are randomized.
        effort_limit_distribution_params: Distribution parameters for the effort limit [N·m].
        velocity_limit_distribution_params: Distribution parameters for the velocity limit [rad/s].
        saturation_effort_distribution_params: Distribution parameters for the saturation effort [N·m].
        operation: How the sampled values combine with the defaults.
        distribution: The distribution the values are sampled from.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=asset.device, dtype=torch.long)

    for actuator in asset.actuators.values():
        if not isinstance(actuator, DCMotor):
            continue

        # ``saturation_effort`` is a python float on the config. Promote it to a per-env,
        # per-joint tensor so it can be randomized; ``_clip_effort`` only uses it in
        # elementwise products, so the broadcast shape is compatible either way.
        if not isinstance(actuator._saturation_effort, torch.Tensor):
            actuator._saturation_effort = torch.full_like(
                actuator.effort_limit, float(actuator._saturation_effort)
            )

        # Cache the pre-randomization values once so repeat calls do not compound.
        if not hasattr(actuator, "_dr_default_effort_limit"):
            actuator._dr_default_effort_limit = actuator.effort_limit.clone()
            actuator._dr_default_velocity_limit = actuator.velocity_limit.clone()
            actuator._dr_default_saturation_effort = actuator._saturation_effort.clone()

        if effort_limit_distribution_params is not None:
            sampled = _sample(
                actuator._dr_default_effort_limit, effort_limit_distribution_params, operation, distribution
            )
            actuator.effort_limit[env_ids] = sampled[env_ids].clamp(min=1e-6)
        if velocity_limit_distribution_params is not None:
            sampled = _sample(
                actuator._dr_default_velocity_limit, velocity_limit_distribution_params, operation, distribution
            )
            actuator.velocity_limit[env_ids] = sampled[env_ids].clamp(min=1e-6)
        if saturation_effort_distribution_params is not None:
            sampled = _sample(
                actuator._dr_default_saturation_effort, saturation_effort_distribution_params, operation, distribution
            )
            actuator._saturation_effort[env_ids] = sampled[env_ids].clamp(min=1e-6)

        # ``_vel_at_effort_lim`` is derived from the three quantities above and is cached at
        # construction time, so it has to be recomputed after any of them changes.
        actuator._vel_at_effort_lim = actuator.velocity_limit * (
            1 + actuator.effort_limit / actuator._saturation_effort
        )
