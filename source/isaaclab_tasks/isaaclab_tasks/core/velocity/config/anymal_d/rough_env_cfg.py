# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


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

# Joint/actuator domain-randomization ranges for
# ``Isaac-Velocity-Rough-AnymalD-JointDR``. Adjust these values to tune the
# training distribution. Values are multiplicative scales around the USD
# defaults, e.g. (0.85, 1.15) means ±15%.
JOINT_DR_ARMATURE_SCALE_RANGE = (0.85, 2.0)
JOINT_DR_STIFFNESS_SCALE_RANGE = (0.85, 2.0)
JOINT_DR_DAMPING_SCALE_RANGE = (0.85, 2.0)

_ANYMAL_ACTUATED_JOINTS = SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True)
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
    """Rough-terrain training configuration with actuator and joint domain randomization."""

    def __post_init__(self):
        super().__post_init__()

        # Randomize physics-level joint armature and controller gains once for
        # each environment at startup. Startup-only DR avoids CPU-side property
        # writes during rollouts and is supported by both PhysX and Newton.
        self.events.randomize_joint_armature = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": _ANYMAL_ACTUATED_JOINTS,
                "operation": "scale",
                "distribution": "uniform",
                "armature_distribution_params": JOINT_DR_ARMATURE_SCALE_RANGE,
            },
        )
        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": _ANYMAL_ACTUATED_JOINTS,
                "operation": "scale",
                "distribution": "uniform",
                "stiffness_distribution_params": JOINT_DR_STIFFNESS_SCALE_RANGE,
                "damping_distribution_params": JOINT_DR_DAMPING_SCALE_RANGE,
            },
        )


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
