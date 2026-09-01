# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 built from a current robot description instead of a superseded one.

The shipped ``G1_CFG`` points at ``g1.usd`` / ``g1_minimal.usd``, and both of those -- in Isaac Sim
6.0 and 6.1 alike -- were generated from ``g1_unitree_deprecated.urdf``. Against every current Unitree
URDF, and against the official MJCF derived from one, that asset is a different robot:

===========================  ==================  ==========================
quantity                     shipped USD         current URDF / MJCF / robot
===========================  ==================  ==========================
hip roll joint origin z      0.0                 -0.030465 m
pelvis to ankle, same pose   0.6865 m            0.7429 m
total mass                   32.24 kg            35.11 kg
driven body joints           23                  29
===========================  ==================  ==========================

Six centimetres of leg is not a modelling detail. The same joint command stands the robot six
centimetres taller, which puts the legs nearer full extension where they have the least leverage --
and a policy trained on the short-legged asset scored 75.6% on the 45-episode hold suite against a
current-geometry MuJoCo, and 0% at zero commanded velocity, versus 97.8% and 100% once the geometry
was matched.

This module uses ``g1_29dof_with_dex3_rev_1_0.usd`` from NVIDIA's ``i4h-asset-catalog``, whose leg
joint origins match ``g1_29dof.urdf`` to 0.000 mm and whose joint limits match the MJCF to 0.0000 rad.

**Two consequences for deployment, both improvements.** The 29 driven body joints are exactly
``G1JointIndex`` -- waist roll and pitch and the wrist pitch and yaw joints exist here and did not in
``g1_minimal.usd``, where deployment had to lock all six. And the joint limits no longer need the
narrowing workaround the old asset required.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

G1_29DOF_USD = os.environ.get(
    "G1_29DOF_USD",
    os.path.expanduser("~/workspace/g1_assets/converted/g1_29dof_locomotion/g1_29dof_locomotion.usda"),
)
"""Converted from ``g1_29dof_with_hand_rev_1_0.urdf``; override with ``G1_29DOF_USD``.

Converted rather than taken prebuilt, and the reason is structural, not cosmetic. The i4h catalog's
``g1_29dof_with_dex3_rev_1_0.usd`` has exactly the right kinematics -- its leg joint origins match the
URDF to 0.000 mm -- but it was built with ``make_instanceable: true`` and its collision geometry ends
up inside instance prototypes three levels below each link. Isaac Lab's contact sensor resolves
shapes one level under the body prim, finds none, and the scene dies at startup with ``sensor shape
expr '[]'``. De-instancing the prebuilt asset does not help; the shapes still sit too deep.

Converting the same URDF with :class:`~isaaclab.sim.converters.UrdfConverter` and
``make_instanceable=False`` puts collision geometry where the sensor looks. Rebuild with::

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    UrdfConverter(UrdfConverterCfg(
        asset_path="~/workspace/g1_assets/g1_29dof_locomotion.urdf",
        usd_dir=os.path.expanduser("~/workspace/g1_assets/converted"),
        fix_base=False, merge_fixed_joints=False,
        force_usd_conversion=True, make_instanceable=False,
    ))
"""

EFFORT_LIMITS = {
    # Per-joint torque ceilings [N·m] from the official MJCF's ctrlrange, which is what the motors
    # actually deliver. The shipped G1_CFG uses one blanket number per group -- 300 for the legs, 300
    # for the arms, 20 for the feet -- and all three are wrong in ways that matter: a policy trained
    # against a 300 N·m hip learns to call for torque the robot cannot produce and the excess is
    # silently clipped, while one trained against a 20 N·m ankle never learns it has 50.
    "hip": 88.0,
    "knee": 139.0,
    "waist_yaw": 88.0,
    "ankle": 50.0,
    "waist_roll_pitch": 50.0,
    "shoulder_elbow": 25.0,
    "wrist_roll": 25.0,
    "wrist_pitch_yaw": 5.0,
    "hand": 2.45,
}
"""Motor torque limits, keyed by the group they apply to."""

G1_29DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_29DOF_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # 0.793 m, not the shipped 0.74: this robot's legs are longer, and spawning it at the old
        # height buries the feet 3 cm in the floor, which the solver answers by launching it upward.
        pos=(0.0, 0.0, 0.793),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            # The old asset called this pair elbow_pitch / elbow_roll; the current one splits them
            # into elbow and wrist_roll, matching the robot's own joint names.
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_hip_.*": EFFORT_LIMITS["hip"],
                ".*_knee_joint": EFFORT_LIMITS["knee"],
                "waist_yaw_joint": EFFORT_LIMITS["waist_yaw"],
            },
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "waist_yaw_joint": 200.0,
            },
            damping={".*_hip_.*": 5.0, ".*_knee_joint": 5.0, "waist_yaw_joint": 5.0},
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=EFFORT_LIMITS["ankle"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "waist": ImplicitActuatorCfg(
            # Absent from the old asset, which forced deployment to lock these two at zero and cost a
            # torso that swung 30 deg relative to the pelvis. Held stiffly, as the real motors are.
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=EFFORT_LIMITS["waist_roll_pitch"],
            stiffness=200.0,
            damping=5.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_.*": EFFORT_LIMITS["shoulder_elbow"],
                ".*_elbow_joint": EFFORT_LIMITS["shoulder_elbow"],
                ".*_wrist_roll_joint": EFFORT_LIMITS["wrist_roll"],
                ".*_wrist_pitch_joint": EFFORT_LIMITS["wrist_pitch_yaw"],
                ".*_wrist_yaw_joint": EFFORT_LIMITS["wrist_pitch_yaw"],
            },
            stiffness=40.0,
            damping=10.0,
            armature=0.01,
        ),
        "hands": ImplicitActuatorCfg(
            # Dex3 fingers. They are a separate device on their own DDS topic, so nothing here is
            # deployed; they exist so the arms carry the right inertia and the policy sees a
            # consistent robot.
            joint_names_expr=[".*_hand_.*_joint"],
            effort_limit_sim=EFFORT_LIMITS["hand"],
            # 40/10, the gains the shipped G1_CFG gives its fingers, not something softer. A finger
            # is light -- armature 0.001, which the armature randomization can halve -- and at
            # stiffness 1 / damping 0.1 the joint is close to both massless and undamped: training
            # produced NaN observations inside the first iteration. Turning the randomization off
            # fixed it, and so did raising these gains; the pair is what breaks it.
            stiffness=40.0,
            damping=10.0,
            armature=0.001,
        ),
    },
)
"""Unitree G1 29-DoF with Dex3 hands, from the i4h asset catalog."""
