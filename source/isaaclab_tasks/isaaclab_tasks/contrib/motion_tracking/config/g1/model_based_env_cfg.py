# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 physical environment for the official model-based learning equations."""

from isaaclab.envs import mdp
from isaaclab.managers import (
    CommandTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.utils import configclass

from ... import observations
from ...g1_tracking_env_cfg import G1_DATA_DIR, G1TrackingEnvCfg
from ...sim_context import randomize_context


@configclass
class ReferenceCommandCfg(CommandTermCfg):
    class_type = "isaaclab_tasks.contrib.motion_tracking.commands:MotionCommand"
    resampling_time_range = (1.0e9, 1.0e9)
    motion_dir: str = str(G1_DATA_DIR / "lafan1_g1")
    motion_shard_index: int = 0
    motion_shard_count: int = 1
    kinematics_path: str = str(G1_DATA_DIR / "g1_description/g1_29dof.xml")
    joint_names: list[str] | None = None
    tracked_bodies: list[str] | None = None
    source_fps: float = 30.0
    context_body: str = "torso_link"
    reset_noise: bool = True
    evaluation: bool = False
    sampling_mode: str = "external"
    friction_bodies: list[str] = [".*ankle_roll_link"]


@configclass
class ReferenceObservationsCfg:
    @configclass
    class StateCfg(ObservationGroupCfg):
        state = ObservationTermCfg(func=observations.full_state)

    @configclass
    class ActorCfg(ObservationGroupCfg):
        sensor = ObservationTermCfg(func=observations.actor_sensor)
        enable_corruption = False

    full_state = StateCfg()
    actor = ActorCfg()


@configclass
class ReferenceRewardsCfg:
    tracking = RewardTermCfg(func=observations.reward, weight=1.0)


@configclass
class ReferenceTerminationsCfg:
    anchor_pos = TerminationTermCfg(func=observations.anchor_height_failure, params={"threshold": 0.25})
    anchor_ori = TerminationTermCfg(func=observations.anchor_orientation_failure, params={"threshold": 0.8})
    motion_end = TerminationTermCfg(func=observations.motion_end, time_out=True)
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class ReferenceEventsCfg:
    context = EventTermCfg(func=randomize_context, mode="startup")


@configclass
class G1ModelBasedEnvCfg(G1TrackingEnvCfg):
    """Native GenCoord task; inherited asset settings are explicit simulator adaptations."""

    observations = ReferenceObservationsCfg()
    rewards = ReferenceRewardsCfg()
    terminations = ReferenceTerminationsCfg()
    events = ReferenceEventsCfg()

    def __post_init__(self):
        self.commands.motion = ReferenceCommandCfg()
        self.actions.joint_pos.class_type = "isaaclab_tasks.contrib.motion_tracking.commands:BiasedJointPositionAction"
        # Four 5 ms integration steps per policy action, matching the reference integrator coefficient.
        self.sim.dt = 0.005
        self.sim.physics.newton_mjwarp.num_substeps = 1
        self.sim.physics.default.num_substeps = 1
        self.decimation = 4
        self.sim.render_interval = 4

    def play_mode(self):
        self.commands.motion.evaluation = True
        self.commands.motion.reset_noise = False
        self.episode_length_s = 1.0e9


@configclass
class G1MaxCoordModelBasedEnvCfg(G1ModelBasedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.full_state.state.func = observations.maximal_state
