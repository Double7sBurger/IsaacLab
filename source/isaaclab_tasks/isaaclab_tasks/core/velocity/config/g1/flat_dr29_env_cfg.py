# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 flat-ground velocity tracking on a current robot description, for sim-to-real.

The same recipe as :mod:`flat_dr_env_cfg` -- a five-frame observation window, no base linear velocity
in the policy group, and randomization confined to the few quantities a robot description cannot pin
down -- rebuilt on :data:`~g1_29dof_asset.G1_29DOF_CFG` instead of the shipped ``G1_CFG``.

**Why a new file rather than an edit.** The asset change renames joints: ``torso_joint`` becomes
``waist_yaw_joint``, ``elbow_pitch``/``elbow_roll`` become ``elbow``/``wrist_roll``, and the
three-finger hand becomes a Dex3. Every reward term and event that names a joint has to move with it,
and the old task has to keep working so its checkpoints stay reproducible.

**What changed relative to the checkpoint currently deployed**

===========================  ==========================  ==============================
                             deployed (2026-08-18)       here
===========================  ==========================  ==============================
robot description            superseded USD, 23 joints   current USD, 29 joints
torque limits                300 / 300 / 20 blanket      88 / 139 / 50 / 25 / 5 per joint
joint friction randomized    no                          yes, 0.0-0.3 N·m
waist roll and pitch         absent, locked on hardware  driven
joint-limit workaround       needed                      unnecessary, limits already match
===========================  ==========================  ==============================

Command latency is deliberately **not** modelled. An earlier variant lagged the leg commands 0-20 ms
through ``DelayedPDActuatorCfg``; that conversion also forces the groups to explicit PD, and the
shipped implicit actuators are what the working checkpoint was trained against. Latency can be added
back once this baseline is established.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import EventsCfg, ObservationsCfg

from .flat_env_cfg import G1FlatEnvCfg
from .g1_29dof_asset import G1_29DOF_CFG

_HISTORY_LENGTH = 5
"""Proprioceptive frames the policy sees, oldest first."""

_ARM_JOINTS = (
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_roll_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_yaw_joint",
)
"""Arm joints under the current naming; the old asset called two of these elbow_pitch / elbow_roll."""

_TORSO_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
"""Waist joints. The old asset had only the yaw, which is why deployment had to lock the other two."""

_BODY_JOINTS = (
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
    *_TORSO_JOINTS,
    *_ARM_JOINTS,
)
"""The 29 motors the robot exposes on ``rt/lowcmd`` -- everything except the Dex3 fingers.

Randomization is scoped to these and not to ``.*`` on purpose. The point of randomizing armature and
actuator stiffness is to cover what a robot description cannot pin down *about the actuators being
deployed*, and the fingers are a separate device on their own DDS topic that this policy never
commands. Including them buys nothing and costs stability: a Dex3 finger link carries 1.5e-06 kg·m^2
of inertia, so its 0.001 armature dominates, and randomizing that down to 0.0005 while randomizing
stiffness up to 80 puts the joint at omega*dt = 2.0. Training NaN'd inside the first iteration until
this scope was narrowed.
"""


@configclass
class G1FlatDR29ObservationsCfg(ObservationsCfg):
    """Stock observations, split so the actor only sees what a real G1 publishes.

    ``base_lin_vel`` moves to a critic-only group: the G1 has no base linear velocity to give, it
    would have to come from a state estimator the robot does not ship, and a policy that leans on it
    in simulation has nothing to read on hardware. The critic runs only in training and is never
    exported, so it keeps the ground truth and the value function stays well conditioned.
    """

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged state, simulation only."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class G1FlatDR29EventsCfg(EventsCfg):
    """Stock events plus the four quantities a robot description cannot pin down.

    Every term is ``startup`` mode: friction is a property of the floor, armature is reflected rotor
    inertia through a gearbox, joint friction is a property of the harmonic drive, and a motor's
    position-loop gain is a property of the unit. None changes between two runs of the same robot on
    the same floor, so each environment is one draw and 4096 environments are 4096 robots. The
    five-frame window is what lets the policy work out which one it is driving.
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            # The stock values are 0.8 static / 0.6 dynamic, so these are those spans scaled 0.5-2x.
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(_BODY_JOINTS)),
            "armature_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            # The USD assumes frictionless joints; a harmonic drive is not, and the official MuJoCo
            # G1 carries 0.2 N·m of frictionloss on every joint. Absolute values, because there is no
            # nonzero nominal to scale, and the span brackets the MJCF's number rather than stopping
            # short of it the way an earlier 0.0-0.05 range did.
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(_BODY_JOINTS)),
            "friction_distribution_params": (0.0, 0.3),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    actuator_stiffness = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            # Stiffness only -- damping stays at nominal so the effective damping ratio moves with
            # the draw, which is what a real position loop does when its gain is off.
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(_BODY_JOINTS)),
            "stiffness_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


@configclass
class G1FlatDR29EnvCfg(G1FlatEnvCfg):
    """Flat G1 locomotion on the current robot description, with a transfer-shaped randomization set."""

    events: G1FlatDR29EventsCfg = G1FlatDR29EventsCfg()
    observations: G1FlatDR29ObservationsCfg = G1FlatDR29ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Not observable on hardware; see G1FlatDR29ObservationsCfg. A term set to None is skipped by
        # the observation manager, the same way the flat parent drops the height scan.
        self.observations.policy.base_lin_vel = None

        # A group-level history overrides every term's, so the policy vector becomes five stacked
        # frames. History is buffered per term and the terms concatenated afterwards, so the layout is
        # term by term -- [ang_vel(t-4..t), gravity(t-4..t), ...] -- not frame by frame. A deployment
        # stack has to reproduce that order, not just the frame count.
        self.observations.policy.history_length = _HISTORY_LENGTH
        self.observations.policy.flatten_history_dim = True

        # Joints the parent task names under the superseded asset's vocabulary.
        self.rewards.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=list(_ARM_JOINTS))
        self.rewards.joint_deviation_torso.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(_TORSO_JOINTS)
        )
        # The three-finger hand became a Dex3; the penalty keeps the same job of holding it still.
        self.rewards.joint_deviation_fingers.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hand_.*_joint"]
        )
