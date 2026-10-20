# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for the Unitree G1 velocity-tracking environment on rough terrain.

Beside the standard environment this module holds the two stages of the depth pipeline: a teacher
whose actor reads a terrain height scan and the base linear velocity, and a student that replaces
both with a chest-mounted depth camera and is distilled from the teacher in the teacher's own
environment.
"""

import math
from collections.abc import Sequence
from copy import deepcopy

import torch
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg, UsdFileCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.images import CameraFrameStack

from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from isaaclab_assets import G1_MINIMAL_CFG

from ... import mdp
from ...velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
)
from . import mdp as g1_mdp


@configclass
class G1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                ],
            )
        },
    )
    joint_deviation_fingers = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_five_joint",
                    ".*_three_joint",
                    ".*_six_joint",
                    ".*_four_joint",
                    ".*_zero_joint",
                    ".*_one_joint",
                    ".*_two_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    )


@configclass
class G1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for the Unitree G1 velocity-tracking environment on rough terrain."""

    rewards: G1Rewards = G1Rewards()

    def __post_init__(self):
        super().__post_init__()

        # physics
        self.sim.physics.newton_mjwarp.solver_cfg.njmax = 300
        # scene
        self.scene.robot = replace(G1_MINIMAL_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        # commands
        self.commands.base_velocity.vel_yaw_success_threshold = 0.8
        self.commands.base_velocity.marker_pos_offset = (0.0, 0.0, 0.75)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # rewards
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
        # events
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.base_com = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "torso_link"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)


DEPTH_WIDTH = 64

DEPTH_HEIGHT = 38

DEPTH_STACK = 3

DEPTH_MAX_RANGE = 3.0

DEPTH_OFFSET_POS = (0.0576, 0.0175, 0.4299)

DEPTH_OFFSET_ROT = (0.9150, 0.0, -0.4035, 0.0)


class DepthImageStack(ManagerTermBase):
    """Stack three D435 depth frames as CNN channels, resetting history with each episode."""

    def __init__(self, cfg: ObsTerm, env) -> None:
        super().__init__(cfg, env)
        self._frames = CameraFrameStack(env.num_envs, env.device, frame_stack=DEPTH_STACK, channel_first=True)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._frames.reset(env_ids)

    def __call__(self, env, sensor_cfg: SceneEntityCfg, data_type: str) -> torch.Tensor:
        frame = env.scene.sensors[sensor_cfg.name].data.output[data_type]
        frame = frame.torch if hasattr(frame, "torch") else frame
        frame = torch.nan_to_num(frame, nan=DEPTH_MAX_RANGE, posinf=DEPTH_MAX_RANGE)
        return self._frames(frame.flip((1, 2)))


G1_29DOF_VELOCITY_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_29dof.usd",
        activate_contact_sensors=True,
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
            ),
            NewtonArticulationCfg(self_collision_enabled=False),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.793),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_joint": 0.87,
            "left_shoulder_pitch_joint": 0.35,
            "left_shoulder_roll_joint": 0.16,
            "right_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_knee_joint", "waist_yaw_joint"],
            joint_effort_limit=300.0,
            stiffness={
                ".*_hip_pitch_joint": 200.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_yaw_joint": 150.0,
                ".*_knee_joint": 200.0,
                "waist_yaw_joint": 200.0,
            },
            damping=5.0,
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_.*"], joint_effort_limit=20.0, stiffness=20.0, damping=2.0, armature=0.01
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            joint_effort_limit=50.0,
            stiffness=200.0,
            damping=5.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"],
            joint_effort_limit=300.0,
            stiffness=40.0,
            damping=10.0,
            armature=0.01,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[".*_hand_.*"], joint_effort_limit=300.0, stiffness=40.0, damping=10.0, armature=0.001
        ),
    },
)


@configclass
class G1RoughTeacherEnvCfg(G1RoughEnvCfg):
    """Height35 teacher on mesh terrain with the original training actuator limits."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = replace(G1_29DOF_VELOCITY_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        self.rewards.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
        )
        self.rewards.joint_deviation_fingers.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hand_.*_joint"]
        )
        self.rewards.joint_deviation_torso.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
        )

        self.seed = 42
        self.scene.num_envs = 4096
        self.sim.use_newton_actuators = False
        self.sim.visualizer_cfgs = []
        self.sim.default_visualizer_cfg = None
        terrain = self.scene.terrain.terrain_generator
        proportions = {
            "pyramid_stairs": 0.2,
            "pyramid_stairs_inv": 0.2,
            "boxes": 0.2,
            "random_rough": 0.2,
            "hf_pyramid_slope": 0.1,
            "hf_pyramid_slope_inv": 0.1,
        }
        for name, subterrain in terrain.sub_terrains.items():
            subterrain.convert_to_heightfield = False
            subterrain.proportion = proportions[name]
        for name in ("pyramid_stairs", "pyramid_stairs_inv"):
            terrain.sub_terrains[name].step_height_range = (0.05, 0.35)
            terrain.sub_terrains[name].step_width = 0.3
        self.actions.joint_pos.scale = {
            ".*_hip_yaw_joint": 0.1467,
            ".*_hip_roll_joint": 0.1467,
            ".*_hip_pitch_joint": 0.11,
            ".*_knee_joint": 0.1738,
            "waist_yaw_joint": 0.11,
            "waist_roll_joint": 0.0625,
            "waist_pitch_joint": 0.0625,
            ".*_ankle_pitch_joint": 0.625,
            ".*_ankle_roll_joint": 0.625,
            ".*_shoulder_.*_joint": 0.1563,
            ".*_elbow_joint": 0.1563,
            ".*_wrist_roll_joint": 0.1563,
            ".*_wrist_pitch_joint": 0.0313,
            ".*_wrist_yaw_joint": 0.0313,
            ".*_hand_.*_joint": 0.0153,
        }
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_slide.weight = -0.5
        self.rewards.joint_deviation_hip.func = g1_mdp.joint_deviation_l2
        self.rewards.joint_deviation_hip.weight = -1.0
        self.rewards.joint_deviation_hip.params["asset_cfg"].joint_names = [".*_hip_roll_joint", ".*_hip_yaw_joint"]
        self.rewards.joint_deviation_torso.func = g1_mdp.joint_deviation_l2
        self.rewards.joint_deviation_torso.weight = -1.0
        self.rewards.pelvis_height = RewTerm(
            func=g1_mdp.pelvis_height_deficit_l2,
            weight=-10.0,
            params={
                "target_height": 0.65,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
        self.rewards.torso_ang_vel = RewTerm(
            func=g1_mdp.body_ang_vel_xy_l2,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])},
        )
        self.rewards.root_downward_speed = RewTerm(
            func=g1_mdp.root_downward_speed_l2,
            weight=-1.0,
            params={"speed_limit": 0.5},
        )
        self.rewards.feet_support_edge = RewTerm(
            func=g1_mdp.feet_support_edge_distance,
            weight=-5.0,
            params={
                **deepcopy(self.rewards.feet_slide.params),
                "sole_size": (0.203109218, 0.065469244, 0.0185078794),
                "sole_offset": (0.0359170487, 0.0, -0.0251700647),
            },
        )
        self.terminations.base_height = DoneTerm(
            func=g1_mdp.pelvis_below_terrain_clearance_after_warmup,
            params={
                "minimum_height": 0.4,
                "warmup_steps": 12000,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )


@configclass
class G1DepthStudentObservationsCfg(ObservationsCfg):
    """Deployable observations, clean privileged targets and an auxiliary reflected camera."""

    teacher: ObservationsCfg.PolicyCfg = ObservationsCfg.PolicyCfg()

    @configclass
    class DepthCfg(ObsGroup):
        depth = ObsTerm(
            func=DepthImageStack,
            params={"sensor_cfg": SceneEntityCfg("depth_camera"), "data_type": "distance_to_image_plane"},
            clip=(0.0, DEPTH_MAX_RANGE),
            scale=1.0 / DEPTH_MAX_RANGE,
        )
        enable_corruption = False
        concatenate_terms = True

    depth: DepthCfg = DepthCfg()
    depth_mirror: DepthCfg = DepthCfg()


@configclass
class G1RoughDepthStudentEnvCfg(G1RoughTeacherEnvCfg):
    """Distill height35 with five proprioceptive frames and mirrored depth supervision."""

    observations: G1DepthStudentObservationsCfg = G1DepthStudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        for name in ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions"):
            term = getattr(self.observations.policy, name)
            term.history_length = 5
            term.flatten_history_dim = True
        self.observations.teacher.enable_corruption = False
        self.scene.depth_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/depth_camera",
            update_period=0.0,
            update_latest_camera_pose=True,
            renderer_cfg=MultiBackendRendererCfg(),
            width=DEPTH_WIDTH,
            height=DEPTH_HEIGHT,
            data_types=["distance_to_image_plane"],
            spawn=PinholeCameraCfg(
                focal_length=1.93,
                horizontal_aperture=2.0 * 1.93 * math.tan(math.radians(87.0 / 2.0)),
                clipping_range=(0.05, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=DEPTH_OFFSET_POS, rot=DEPTH_OFFSET_ROT, convention="world"),
        )
        self.scene.depth_camera_mirror = deepcopy(self.scene.depth_camera)
        self.scene.depth_camera_mirror.prim_path = "{ENV_REGEX_NS}/Robot/torso_link/depth_camera_mirror"
        x, y, z = DEPTH_OFFSET_POS
        self.scene.depth_camera_mirror.offset.pos = (x, -y, z)
        self.observations.depth_mirror.depth.params["sensor_cfg"] = SceneEntityCfg("depth_camera_mirror")
