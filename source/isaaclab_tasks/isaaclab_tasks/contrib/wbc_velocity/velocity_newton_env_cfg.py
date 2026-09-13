# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WBC-AGILE's G1 velocity task on the Newton/MJWarp backend.

WBC only ever runs this task on PhysX: ``velocity_env_cfg`` forces ``PhysxCfg()`` and sets a PhysX
GPU buffer size. This is the same MDP with the physics backend swapped and nothing else changed, so
that a Newton run is comparable term-for-term with the PhysX one.

Three things that look like they would need changing and do not:

* **The USD variant.** ``variants={"Physics": "PhysX"}`` selects the *authoring* variant of
  ``g1.usd``; the variant set offers only ``None``/``PhysX``/``SimplifiedPhysX`` and has no Newton
  option, because Newton consumes the same UsdPhysics schemas that variant writes.
* **The actuators.** ``sim.use_newton_actuators`` stays ``False``. Newton-native actuator execution
  cannot run ``DelayedDCMotorCfg``, and the delay is part of what is being reproduced, so the
  Lab-side actuator path is used on both backends.
* **The terrain.** ``LESS_ROUGH_TERRAIN_CFG``'s ``Hf*`` sub-terrains are generated as height fields
  but converted to trimesh, which is what keeps this off the height-field collider path.
"""

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.wbc_velocity.velocity_env_cfg import G1LowerVelocityEnvCfg
from isaaclab_tasks.contrib.wbc_velocity.velocity_history_env_cfg import G1LowerVelocityHistoryEnvCfg


def _newton_mjwarp_cfg() -> NewtonCfg:
    """Isaac Lab's own tuned MJWarp preset for locomotion velocity tasks.

    Taken verbatim from :class:`~isaaclab_tasks.core.velocity.velocity_env_cfg.RoughPhysicsCfg` so
    this arm is not also an untuned-solver experiment. WBC never picked Newton numbers, so the
    alternative would be inventing them.
    """
    return NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=1000,
            nconmax=300,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=2,
        debug_mode=False,
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
    )


@configclass
class G1LowerVelocityNewtonEnvCfg(G1LowerVelocityEnvCfg):
    """The WBC G1 velocity task, on Newton/MJWarp instead of PhysX."""

    def __post_init__(self):
        super().__post_init__()
        # The parent installs PhysxCfg() and a PhysX-only GPU buffer size; replace the whole object
        # rather than editing it, so no PhysX field survives onto the Newton config.
        self.sim.physics = _newton_mjwarp_cfg()


@configclass
class G1LowerVelocityHistoryNewtonEnvCfg(G1LowerVelocityHistoryEnvCfg):
    """The history-observation variant, on Newton/MJWarp."""

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = _newton_mjwarp_cfg()
