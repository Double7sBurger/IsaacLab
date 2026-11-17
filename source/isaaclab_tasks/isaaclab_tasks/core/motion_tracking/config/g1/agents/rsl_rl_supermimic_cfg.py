# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GenCoord motion-tracking learner with the SuperMimic network and rollout settings."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg
from isaaclab_rl.rsl_rl.supermimic_cfg import (
    RslRlSuperMimicActorCfg,
    RslRlSuperMimicAlgorithmCfg,
    RslRlSuperMimicRunnerCfg,
    RslRlSuperMimicTaskCfg,
    RslRlWorldModelCfg,
)


@configclass
class MotionTrackingCriticCfg(RslRlMLPModelCfg):
    hidden_dims = [256, 128]
    activation = "gelu"
    obs_normalization = False
    network_cfg = {"class_name": "MLP", "hidden_dims": [256, 128], "activation": "gelu", "norm": "layer"}


@configclass
class G1MotionTrackingSuperMimicRunnerCfg(RslRlSuperMimicRunnerCfg):
    """One physical step per iteration, with replay-based differentiable policy updates."""

    num_steps_per_env = 1
    max_iterations = 320_000
    save_interval = 4000
    experiment_name = "g1_motion_tracking_supermimic"
    clip_actions = None
    obs_groups = {"state": ["state"], "goal": ["goal"], "context": ["context"], "prev_action": ["prev_action"]}
    actor = RslRlSuperMimicActorCfg(
        history_length=10, activation="gelu", norm="layer", output_gain=1.0, normalize_before_action=False
    )
    world_model = RslRlWorldModelCfg(
        activation="gelu", normalize_output=False, normalize_context=False, zero_initialize_output=False
    )
    critic = MotionTrackingCriticCfg()
    algorithm = RslRlSuperMimicAlgorithmCfg(
        task_cfg=RslRlSuperMimicTaskCfg(class_name="isaaclab_tasks.core.motion_tracking.supermimic:MotionTrackingTask"),
        replay_buffer_capacity=4_096_000,
        world_model_window=8,
        world_model_loss_reduction="mean",
        world_model_full_windows=True,
        num_warmup_steps=9,
        num_world_model_warmup_iterations=0,
        update_schedule="from_start",
        final_learning_rate_ratio=0.03,
        learning_rate_schedule_unit="env_steps",
        critic_survival_weighting=False,
        critic_bootstrap_clipping=False,
        critic_shuffle_each_epoch=False,
        max_step_reward=0.1,
        use_cuda_graph=True,
    )
