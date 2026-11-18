# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Model-based G1 tracking using the shared motion-tracking asset and motor configuration."""

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass, replace

from ... import supermimic
from ...mdp.supermimic_commands import BiasedJointPositionAction, SuperMimicMotionCommandCfg
from .tracking_env_cfg import G1ActionsCfg, G1CommandsCfg, G1MotionTrackingEnvCfg, G1SceneCfg


@configclass
class SuperMimicObservationsCfg:
    """Noise-free physical replay groups; the actor synthesizes its noisy sensor history."""

    @configclass
    class StateCfg(ObservationGroupCfg):
        state = ObservationTermCfg(func=supermimic.physical_state)

    @configclass
    class GoalCfg(ObservationGroupCfg):
        phase = ObservationTermCfg(func=supermimic.reference_phase)

    @configclass
    class ContextCfg(ObservationGroupCfg):
        context = ObservationTermCfg(func=supermimic.physical_context)

    @configclass
    class ActionCfg(ObservationGroupCfg):
        action = ObservationTermCfg(func=mdp.last_action)

    state = StateCfg()
    goal = GoalCfg()
    context = ContextCfg()
    prev_action = ActionCfg()


@configclass
class SuperMimicRewardsCfg:
    tracking = RewardTermCfg(func=supermimic.tracking_reward, weight=1.0)


@configclass
class SuperMimicTerminationsCfg:
    anchor_pos = TerminationTermCfg(func=supermimic.height_failure)
    anchor_ori = TerminationTermCfg(func=supermimic.orientation_failure)
    motion_end = TerminationTermCfg(func=supermimic.motion_finished, time_out=True)
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class SuperMimicCommandsCfg:
    motion = SuperMimicMotionCommandCfg(
        **{key: value for key, value in vars(G1CommandsCfg().motion).items() if key != "class_type"}
    )
    motion.anchor_body_name = "pelvis"


@configclass
class SuperMimicSceneCfg(G1SceneCfg):
    self_collision = None


@configclass
class SuperMimicActionsCfg(G1ActionsCfg):
    joint_pos = replace(G1ActionsCfg().joint_pos, class_type=BiasedJointPositionAction)


@configclass
class SuperMimicEventsCfg:
    """Physical context randomization is applied when the motion command resets."""


@configclass
class G1SuperMimicEnvCfg(G1MotionTrackingEnvCfg):
    """Track reference windows with an independently trained world model and BPTT actor."""

    scene = SuperMimicSceneCfg(num_envs=4096, env_spacing=2.5)
    actions = SuperMimicActionsCfg()
    commands = SuperMimicCommandsCfg()
    observations = SuperMimicObservationsCfg()
    rewards = SuperMimicRewardsCfg()
    terminations = SuperMimicTerminationsCfg()
    events = SuperMimicEventsCfg()
    final_obs_every_step = True

    def play_mode(self):
        self.commands.motion.sampling_mode = "start"
        self.commands.motion.randomize_context = False
        self.commands.motion.pose_range = {}
        self.commands.motion.velocity_range = {}
        self.commands.motion.joint_position_range = (0.0, 0.0)
