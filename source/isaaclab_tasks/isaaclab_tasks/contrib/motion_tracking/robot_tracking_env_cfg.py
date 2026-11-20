# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion tracking with the assets, actuators and timing of the native flat velocity tasks."""

from isaaclab.paths import ISAACLAB_ROOT
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg
from isaaclab_tasks.core.velocity.config.cassie.flat_env_cfg import CassieFlatEnvCfg
from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.core.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg

from .g1_tracking_env_cfg import G1TrackingEnvCfg


def _use_robot(cfg, native, robot, joints, bodies):
    cfg.scene.robot = native.scene.robot.copy()
    cfg.scene.ground.spawn.physics_material = native.scene.terrain.physics_material.copy()
    cfg.sim = native.sim.copy()
    cfg.decimation = native.decimation
    cfg.actions.joint_pos = native.actions.joint_pos.copy()
    cfg.actions.joint_pos.class_type = "isaaclab_tasks.contrib.motion_tracking.mdp:ContextJointPositionAction"
    cfg.commands.motion.joint_names = joints
    cfg.commands.motion.tracked_bodies = bodies
    cfg.commands.motion.context_body = bodies[0]
    # Scale the fall threshold with robot size.
    cfg.commands.motion.objective.term_height = min(0.25, 0.25 * native.scene.robot.init_state.pos[2])
    cfg.commands.motion.motion_dir = f"{ISAACLAB_ROOT}/data/four_robot_tracking/{robot}/motions"
    cfg.commands.motion.kinematics_path = f"{ISAACLAB_ROOT}/data/four_robot_tracking/{robot}/robot.xml"


@configclass
class AnymalDTrackingEnvCfg(G1TrackingEnvCfg):
    def __post_init__(self):
        _use_robot(
            self,
            AnymalDFlatEnvCfg(),
            "AnymalD",
            [
                "LF_HAA",
                "LF_HFE",
                "LF_KFE",
                "LH_HAA",
                "LH_HFE",
                "LH_KFE",
                "RF_HAA",
                "RF_HFE",
                "RF_KFE",
                "RH_HAA",
                "RH_HFE",
                "RH_KFE",
            ],
            [
                "base",
                "LF_HIP",
                "LF_THIGH",
                "LF_SHANK",
                "LF_FOOT",
                "LH_HIP",
                "LH_THIGH",
                "LH_SHANK",
                "LH_FOOT",
                "RF_HIP",
                "RF_THIGH",
                "RF_SHANK",
                "RF_FOOT",
                "RH_HIP",
                "RH_THIGH",
                "RH_SHANK",
                "RH_FOOT",
            ],
        )


@configclass
class CassieTrackingEnvCfg(G1TrackingEnvCfg):
    def __post_init__(self):
        _use_robot(
            self,
            CassieFlatEnvCfg(),
            "Cassie",
            [
                "hip_abduction_left",
                "hip_rotation_left",
                "hip_flexion_left",
                "thigh_joint_left",
                "ankle_joint_left",
                "toe_joint_left",
                "hip_abduction_right",
                "hip_rotation_right",
                "hip_flexion_right",
                "thigh_joint_right",
                "ankle_joint_right",
                "toe_joint_right",
            ],
            [
                "pelvis",
                "left_pelvis_rotation",
                "left_hip",
                "left_thigh",
                "left_shin",
                "left_tarsus",
                "left_toe",
                "right_pelvis_rotation",
                "right_hip",
                "right_thigh",
                "right_shin",
                "right_tarsus",
                "right_toe",
            ],
        )


@configclass
class UnitreeGo2TrackingEnvCfg(G1TrackingEnvCfg):
    def __post_init__(self):
        _use_robot(
            self,
            UnitreeGo2FlatEnvCfg(),
            "UnitreeGo2",
            [
                "FL_hip_joint",
                "FL_thigh_joint",
                "FL_calf_joint",
                "FR_hip_joint",
                "FR_thigh_joint",
                "FR_calf_joint",
                "RL_hip_joint",
                "RL_thigh_joint",
                "RL_calf_joint",
                "RR_hip_joint",
                "RR_thigh_joint",
                "RR_calf_joint",
            ],
            [
                "base",
                "FL_hip",
                "FL_thigh",
                "FL_calf",
                "FL_foot",
                "FR_hip",
                "FR_thigh",
                "FR_calf",
                "FR_foot",
                "Head_upper",
                "Head_lower",
                "RL_hip",
                "RL_thigh",
                "RL_calf",
                "RL_foot",
                "RR_hip",
                "RR_thigh",
                "RR_calf",
                "RR_foot",
            ],
        )


@configclass
class H1TrackingEnvCfg(G1TrackingEnvCfg):
    def __post_init__(self):
        _use_robot(
            self,
            H1FlatEnvCfg(),
            "H1",
            [
                "left_hip_yaw",
                "left_hip_roll",
                "left_hip_pitch",
                "left_knee",
                "left_ankle",
                "right_hip_yaw",
                "right_hip_roll",
                "right_hip_pitch",
                "right_knee",
                "right_ankle",
                "torso",
                "left_shoulder_pitch",
                "left_shoulder_roll",
                "left_shoulder_yaw",
                "left_elbow",
                "right_shoulder_pitch",
                "right_shoulder_roll",
                "right_shoulder_yaw",
                "right_elbow",
            ],
            [
                "pelvis",
                "left_hip_yaw_link",
                "left_hip_roll_link",
                "left_hip_pitch_link",
                "left_knee_link",
                "left_ankle_link",
                "right_hip_yaw_link",
                "right_hip_roll_link",
                "right_hip_pitch_link",
                "right_knee_link",
                "right_ankle_link",
                "torso_link",
                "left_shoulder_pitch_link",
                "left_shoulder_roll_link",
                "left_shoulder_yaw_link",
                "left_elbow_link",
                "right_shoulder_pitch_link",
                "right_shoulder_roll_link",
                "right_shoulder_yaw_link",
                "right_elbow_link",
            ],
        )
