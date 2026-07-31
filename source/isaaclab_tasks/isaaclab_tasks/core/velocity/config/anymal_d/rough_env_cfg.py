# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab_newton.physics import NewtonShapeCfg
from isaaclab_visualizers.newton import NewtonVisualizerCfg

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip
import isaaclab.sim as sim_utils

# Joint/actuator domain-randomization ranges for ``Isaac-Velocity-Rough-AnymalD-JointDR``.
#
# IMPORTANT -- these are ABSOLUTE values, not multiplicative scales. ANYmal-D's USD ships
# with armature = 0 and joint friction = 0, so a "scale" operation samples a random number,
# multiplies zero by it, and writes zero back: the randomization is a no-op that only
# perturbs the RNG stream. Any quantity whose default is zero must be randomized with
# ``operation="abs"`` (or ``"add"``).
#
# Note that stiffness/damping are deliberately absent. ANYmal-D is driven by
# ``ANYDRIVE_3_LSTM_ACTUATOR_CFG`` (:class:`~isaaclab.actuators.ActuatorNetLSTM`), which
# computes torque from a learned network and never reads either gain. Randomizing them
# cannot change the dynamics; the live actuation knob is the DC-motor torque-speed clip,
# randomized by :func:`mdp.randomize_dc_motor_limits` below.

# Absolute joint armature [kg·m²]. ANYdrive 3 reflects its rotor inertia through a ~50:1
# gearbox, so the physically plausible band is roughly 1e-2 .. 1e-1. The lower bound stays
# near the USD default of 0 so the regime the current checkpoints were trained in remains
# inside the distribution.
JOINT_DR_ARMATURE_RANGE = (0.005, 0.05)

# Absolute joint Coulomb friction [N·m].
JOINT_DR_FRICTION_RANGE = (0.0, 0.05)

# Multiplicative scales on the DC-motor torque-speed envelope. These defaults are non-zero
# (effort_limit=80 N·m, velocity_limit=7.5 rad/s, saturation_effort=120 N·m), so "scale" is
# meaningful here.
JOINT_DR_EFFORT_LIMIT_SCALE_RANGE = (0.8, 1.2)
JOINT_DR_VELOCITY_LIMIT_SCALE_RANGE = (0.9, 1.1)
JOINT_DR_SATURATION_EFFORT_SCALE_RANGE = (0.8, 1.2)

# Foot/body contact friction. The stock ``physics_material`` term pins these to single
# points -- (0.8, 0.8) static and (0.6, 0.6) dynamic -- i.e. no randomization at all.
# Contact friction is the largest behavioural difference between the PhysX and MJWarp
# contact models, so it is the highest-value quantity to randomize for sim2sim.
#
# Beware an asymmetry: PhysX honours both coefficients, while Newton has a single friction
# coefficient and takes it from ``static_friction_range``, ignoring the dynamic range
# entirely (see ``_RandomizeRigidBodyMaterialNewton``). The two backends therefore do NOT
# see the same nominal friction under the stock config.
JOINT_DR_STATIC_FRICTION_RANGE = (0.6, 1.2)
JOINT_DR_DYNAMIC_FRICTION_RANGE = (0.4, 1.0)

_ANYMAL_ACTUATED_JOINTS = SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True)

# Terrain seed for the deterministic comparison configs.
#
# ``TerrainGeneratorCfg.seed`` defaults to None, in which case TerrainGenerator falls back to
# ``np.random.get_state()[1][0]`` (see ``terrain_generator.py``) -- the ambient NumPy state at
# construction time. The ``-Compare-Play`` and ``-SysID-Play`` configs pin it below; the plain
# ``-PhysxOrder-Play`` config deliberately does not, so its historical numbers stay valid.
#
# To pin it there too, override on the command line rather than editing this file:
#     env.scene.terrain.terrain_generator.seed=42
_COMPARE_TERRAIN_SEED = 42


@configclass
class AnymalDRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-d
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class AnymalDRoughEnvCfg_JOINT_DR(AnymalDRoughEnvCfg):
    """Rough-terrain training configuration with actuator and joint domain randomization.

    Every term here is randomized at startup. Startup-only DR avoids CPU-side property
    writes during rollouts and is supported by both PhysX and Newton.

    To verify the randomization is live rather than silently degenerate, check that the
    sampled quantities differ across environments after construction -- a "scale" operation
    applied to a default of zero produces an environment identical to
    :class:`AnymalDRoughEnvCfg`.
    """

    def __post_init__(self):
        super().__post_init__()

        # Physics-level joint properties. Both default to zero in the USD, so the operation
        # must be "abs": the sampled value replaces the default rather than scaling it.
        self.events.randomize_joint_armature = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": _ANYMAL_ACTUATED_JOINTS,
                "operation": "abs",
                "distribution": "uniform",
                "armature_distribution_params": JOINT_DR_ARMATURE_RANGE,
                "friction_distribution_params": JOINT_DR_FRICTION_RANGE,
            },
        )

        # The live actuation knob for the LSTM actuator net: the DC-motor torque-speed clip
        # that bounds whatever torque the network produces.
        self.events.randomize_actuator_limits = EventTerm(
            func=mdp.randomize_dc_motor_limits,
            mode="startup",
            params={
                "asset_cfg": _ANYMAL_ACTUATED_JOINTS,
                "operation": "scale",
                "distribution": "uniform",
                "effort_limit_distribution_params": JOINT_DR_EFFORT_LIMIT_SCALE_RANGE,
                "velocity_limit_distribution_params": JOINT_DR_VELOCITY_LIMIT_SCALE_RANGE,
                "saturation_effort_distribution_params": JOINT_DR_SATURATION_EFFORT_SCALE_RANGE,
            },
        )

        # Widen the contact friction the base config pins to a single point.
        self.events.physics_material.params["static_friction_range"] = JOINT_DR_STATIC_FRICTION_RANGE
        self.events.physics_material.params["dynamic_friction_range"] = JOINT_DR_DYNAMIC_FRICTION_RANGE
        self.events.physics_material.params["make_consistent"] = True


@configclass
class AnymalDRoughEnvCfg_PLAY(AnymalDRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Newton's interactive camera is static. Its tiled camera follows the robot
        # and renders from Newton's simulated articulation state for playback videos.
        self.sim.visualizer_cfgs = NewtonVisualizerCfg(
            window_width=1280,
            window_height=720,
            tiled_cam_view=True,
            tiled_cam_num=1,
            tiled_cam_env_indices=[0],
            tiled_cam_target_prim_path="/World/envs/*/Robot",
            tiled_cam_eye=(2.5, 2.5, 1.5),
        )


@configclass
class AnymalDRoughEnvCfg_PHYSX_ORDERED_PLAY(AnymalDRoughEnvCfg_PLAY):
    """Playback configuration that preserves a PhysX policy's articulation axes.

    Use only when evaluating a checkpoint trained with PhysX under the Newton
    MJWarp backend. Native Newton ordering remains available internally, while
    the public joint and body tensors used by actions and observations are
    reordered to PhysX convention.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = self.scene.robot.replace(
            joint_ordering="physx",
            body_ordering="physx",
        )


##
# Base collider override
##

# Radius [m] of the base capsule used by ``AnymalDRoughEnvCfg_PHYSX_ORDERED_FIXED_BASE_PLAY``.
#
# The shipped ANYmal-D asset gives the base a capsule collider of radius 0.12 m. Being
# rotationally symmetric it bulges 0.12 m in every direction perpendicular to its X axis,
# which brings it to within a few millimetres of the thigh boxes during normal gait. PhysX
# tolerates that grazing contact; MJWarp answers it with hundreds of newtons on the base,
# tripping the 1 N ``base_contact`` termination and costing ~3 points of success rate.
#
# Measured on ``physx_2000/model_900.pt`` (4096 envs x 3000 steps):
#     r = 0.12 -> Newton 93.06%, PhysX 96.19%   (gap 3.13)
#     r = 0.10 -> Newton 95.55%, PhysX 96.13%   (gap 0.58)
# Upright false-positive failures drop 273 -> 56 while genuine falls stay flat (393 -> 384).
FIXED_BASE_CAPSULE_RADIUS = 0.10

# Newton shape contact margin [m] paired with the shrunk capsule.
#
# These two settings are coupled and must be changed together. Margin inflates *every* shape,
# so with the stock 0.12 m capsule -- which already passes within ~3 mm of the thigh boxes --
# adding margin turns grazing contact into constant overlap and success rate collapses. Once
# the capsule is slim enough to clear the thighs, the same margin delivers what it is meant
# for: contact tolerance between the feet and the triangle-mesh terrain.
#
# Measured on ``physx_2000/model_900.pt`` under Newton (4096 envs x 3000 steps):
#
#     margin |  r = 0.12  |  r = 0.10
#     -------+------------+-----------
#      0     |   93.11%   |   95.58%
#      0.005 |   71.21%   |   96.14%   <- this config
#      0.01  |   28.89%   |   93.77%
#
# PhysX reference: 96.19%. At r=0.10 / margin=0.005 the sim2sim gap is 0.05 points, i.e.
# within the +/-0.2 point sampling error. Genuine falls also drop 374 -> 284 versus margin=0,
# which is the feet no longer catching on the terrain.
FIXED_BASE_SHAPE_MARGIN = 0.005


def _resolve_base_collider_override() -> str:
    """Locate the USD layer that overrides the base capsule radius.

    The layer is generated by ``tools/eval_tools/make_base_collider_override.py`` rather than
    authored here: writing USD inside a process that later boots Kit corrupts Kit's ``pxr``
    bindings. Set ``ANYMAL_D_BASE_OVERRIDE_USD`` to point somewhere other than the default
    location under the repository.

    Raises:
        FileNotFoundError: If the layer has not been generated yet.
    """
    import os
    from pathlib import Path

    explicit = os.environ.get("ANYMAL_D_BASE_OVERRIDE_USD")
    if explicit:
        path = Path(explicit)
    else:
        # .../source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/anymal_d/<this file>
        repo_root = Path(__file__).resolve().parents[7]
        path = (
            repo_root
            / "tools"
            / "eval_tools"
            / "usd_overrides"
            / f"anymal_d_base_capsule_r{FIXED_BASE_CAPSULE_RADIUS:g}.usd"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"The base-collider override layer is missing: {path}\n"
            "Generate it once with:\n"
            "    ./isaaclab.sh -p tools/eval_tools/make_base_collider_override.py"
            f" --radius {FIXED_BASE_CAPSULE_RADIUS:g}\n"
            "Or set ANYMAL_D_BASE_OVERRIDE_USD to an existing layer."
        )
    return str(path)


@configclass
class AnymalDRoughEnvCfg_PHYSX_ORDERED_FIXED_BASE_PLAY(AnymalDRoughEnvCfg_PHYSX_ORDERED_PLAY):
    """PhysX-ordered playback with the Newton sim2sim gap closed.

    Two coupled changes relative to :class:`AnymalDRoughEnvCfg_PHYSX_ORDERED_PLAY`:

    1. The robot spawns from a thin USD layer overriding the base capsule radius to
       :data:`FIXED_BASE_CAPSULE_RADIUS`. The shipped asset is never modified -- the layer
       sublayers it and overrides one attribute.
    2. The Newton shape margin is raised to :data:`FIXED_BASE_SHAPE_MARGIN`, restoring the
       foot/terrain contact tolerance that PhysX gets from ``contact_offset`` (a property the
       Newton backend ignores entirely).

    Neither change works alone -- see the table on :data:`FIXED_BASE_SHAPE_MARGIN`.

    The USD layer must be generated first; see :func:`_resolve_base_collider_override`.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = self.scene.robot.replace(
            spawn=self.scene.robot.spawn.replace(usd_path=_resolve_base_collider_override()),
        )
        # Assign a fresh NewtonShapeCfg rather than writing ``.margin`` on the existing one:
        # the latter would mutate an object the physics preset may share with other configs.
        # (``NewtonCfg.replace()`` is not usable here -- it rejects the auto-derived
        # ``class_type`` field.) Verified not to leak into sibling tasks.
        #
        # ``sim.physics`` is either the still-unresolved preset (with a ``newton_mjwarp``
        # branch) or the already-resolved backend config, depending on when the preset is
        # applied. Handle both; under PhysX neither branch exists and the margin is skipped,
        # which is correct because PhysX has no such setting.
        for candidate in (self.sim.physics, getattr(self.sim.physics, "newton_mjwarp", None)):
            if candidate is not None and hasattr(candidate, "default_shape_cfg"):
                candidate.default_shape_cfg = NewtonShapeCfg(margin=FIXED_BASE_SHAPE_MARGIN)


@configclass
class AnymalDRoughEnvCfg_PHYSX_ORDERED_COMPARE_PLAY(AnymalDRoughEnvCfg_PHYSX_ORDERED_PLAY):
    """Deterministic PhysX/Newton playback configuration for policy comparison.

    This fixes reset state, terrain level, and the velocity command so the two
    backends start from equivalent task inputs. The trajectories can still
    diverge after the first step because their physics solvers differ.
    """

    def __post_init__(self):
        super().__post_init__()

        # Always use the first terrain level rather than randomly selecting a
        # terrain origin as the regular play configuration does.
        self.scene.terrain.max_init_terrain_level = 0
        if self.scene.terrain.terrain_generator is not None:
            # TerrainGeneratorCfg otherwise derives its seed from the current
            # NumPy RNG state. Backend initialization can consume that state
            # differently, yielding distinct static terrain meshes.
            self.scene.terrain.terrain_generator.seed = _COMPARE_TERRAIN_SEED

        # Disable all startup randomization. In particular, PhysX supports COM
        # randomization while Newton disables it, which otherwise advances the
        # two backends' random-number streams by different amounts.
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None

        # Reset every episode to the same root state and nominal joint pose.
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        # Use a constant, non-zero forward command. This avoids a random
        # standing command and makes velocity tracking directly comparable.
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)



        # Since there's penetration issues, we need to increase the solver iteration count and max depenetration velocity
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 1
        self.scene.robot.spawn.rigid_props.max_depenetration_velocity = 3.0
        self.scene.robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        )


# PACE SysID joint order (matches data collection order)
_PACE_JOINT_ORDER = [
    "LF_HAA", "LF_HFE", "LF_KFE",
    "RF_HAA", "RF_HFE", "RF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE",
]

_PACE_CHECKPOINT = (
    "/home/henry/workspace/pace-sim2real/logs/pace/anymal_d_sim/26_07_28_15-43-36/mean_042.pt"
)


@configclass
class AnymalDRoughEnvCfg_NEWTON_SYSID_PLAY(AnymalDRoughEnvCfg_PHYSX_ORDERED_COMPARE_PLAY):
    """Newton play config with PACE-identified armature and friction applied at startup.

    Inherits the deterministic PhysX-ordered compare configuration and adds a
    startup event that writes the PACE SysID parameters (armature + Coulomb friction)
    into Newton. Use this to measure how much the identified parameters close the
    PhysX→Newton sim2sim gap compared to the baseline Newton run.
    """

    def __post_init__(self):
        super().__post_init__()

        self.events.apply_pace_sysid = EventTerm(
            func=mdp.apply_pace_newton_params,
            mode="startup",
            params={
                "pace_checkpoint": _PACE_CHECKPOINT,
                "joint_order": _PACE_JOINT_ORDER,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )