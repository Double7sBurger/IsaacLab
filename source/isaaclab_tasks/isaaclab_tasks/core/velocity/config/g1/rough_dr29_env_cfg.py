# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 rough-terrain velocity tracking on a current robot description, for sim-to-real.

:mod:`flat_dr29_env_cfg` on rough terrain. The randomization set is not re-derived here -- it is
:class:`~flat_dr29_env_cfg.G1FlatDR29EventsCfg`, imported directly, so the two tasks cannot drift
apart: the same four startup-mode terms (ground friction, joint armature, joint friction, actuator
stiffness), the same per-joint torque ceilings from the asset, and the same deliberate absence of
command latency.

**What rough terrain adds.** A height scan. The G1 does not publish one, so it goes to the critic
alongside ``base_lin_vel`` rather than to the policy, following :mod:`rough_dr_env_cfg`. That keeps
the actor's observation identical in shape to the flat DR29 task -- five frames of proprioception,
no privileged state -- so a checkpoint from either task has the same deployment contract, and the
value function still gets the terrain it needs to be well conditioned.

**What this task does not inherit from** :mod:`rough_dr_env_cfg`. That variant also delays the leg
commands and randomizes per-link mass, and its actuator re-typing forces the groups to explicit PD.
DR29 dropped both on purpose: the working checkpoint was trained against implicit actuators, and
latency is to be added back only once a baseline exists. Same decision here.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import ObservationsCfg

from .flat_dr29_env_cfg import _ARM_JOINTS, _HISTORY_LENGTH, _TORSO_JOINTS, G1FlatDR29EventsCfg
from .g1_29dof_asset import G1_29DOF_CFG
from .rough_env_cfg import G1RoughEnvCfg

_TORSO_LINK_PATH = ".*torso_link"
"""How to reach ``torso_link`` from the articulation root, at whatever depth it sits.

The parent task hard-codes ``Robot/torso_link``, which assumes the flat link layout of the shipped
asset. ``UrdfConverter`` preserves the URDF's parent-child chain instead, so on a converted asset
the body is five levels down and the stock pattern silently matches nothing -- the scene then dies
with ``Site 'ft_0' ... matched no source-builder bodies``, which does not mention depth. Matching at
any depth works for both, and both assets contain exactly one body named ``torso_link``.
"""


@configclass
class G1RoughDR29ObservationsCfg(ObservationsCfg):
    """Stock observations, split so the actor only sees what a real G1 publishes.

    ``base_lin_vel`` would have to come from a state estimator the robot does not ship, and
    ``height_scan`` from an elevation-mapping stack it does not run. Both stay in a critic-only
    group, which is never exported. If a mapping stack does become available, move ``height_scan``
    back to the policy group and give it a noise term -- a clean scan in training and a noisy one on
    hardware is its own transfer gap.
    """

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged state, simulation only."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class G1RoughDR29EnvCfg(G1RoughEnvCfg):
    """Rough G1 locomotion on the current robot description, with the flat task's randomization."""

    events: G1FlatDR29EventsCfg = G1FlatDR29EventsCfg()
    observations: G1RoughDR29ObservationsCfg = G1RoughDR29ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # The parent attaches the height scanner to "{ENV_REGEX_NS}/Robot/torso_link", which assumes
        # the shipped asset's flat link layout. This asset is a URDF conversion that keeps the
        # kinematic hierarchy, so the body sits several levels down and the pattern matches nothing:
        # the scene dies with "Site 'ft_0' ... matched no source-builder bodies". The flat task never
        # hit this because it drops the height scanner entirely.
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + _TORSO_LINK_PATH

        # Not observable on hardware; both live on the critic instead. A term set to None is skipped
        # by the observation manager, the same way the flat parent drops the height scan.
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None

        # A group-level history overrides every term's, so the policy vector becomes five stacked
        # frames. History is buffered per term and the terms concatenated afterwards, so the layout
        # is term by term -- [ang_vel(t-4..t), gravity(t-4..t), ...] -- not frame by frame. A
        # deployment stack has to reproduce that order, not just the frame count.
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
