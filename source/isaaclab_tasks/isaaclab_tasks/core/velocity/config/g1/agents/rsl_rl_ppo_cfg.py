# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.utils import preset


@configclass
class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    # Newton needs ~1.7x the PPO iterations to match PhysX on G1. PhysX saturates near iter 3000
    # (reward ≈ +18, ep_len ≈ 980) and does not meaningfully improve on either metric past that —
    # reward oscillates +16 to +19 through iter 7500, ep_len stays flat. Newton reaches the same
    # (reward, ep_len) quality at iter 5000 (+16 / 984). Comparing reward alone is misleading:
    # ep_len confirms the robot is stable in both cases. The gap is sample-efficiency, not a
    # ceiling — no physics or reward tuning closes it.
    max_iterations = preset(default=3000, newton_mjwarp=5000)
    save_interval = 50
    experiment_name = "g1_rough"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1FlatPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "g1_flat"
        self.actor.hidden_dims = [256, 128, 128]
        self.critic.hidden_dims = [256, 128, 128]


@configclass
class G1RoughDRPPORunnerCfg(G1RoughPPORunnerCfg):
    """Runner for the randomized, partially observable G1 rough-terrain task."""

    # The actor sees five frames of noisy proprioception; the critic additionally gets the base
    # linear velocity and the height scan, which is what keeps the value function usable while the
    # actor stays deployable.
    obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}
    # Roughly double the baseline. Two things cost iterations here: the actor is blind on rough
    # terrain, and the dynamics it is fitting move every episode. Treat these as a starting budget,
    # not a converged number -- watch episode length rather than reward, since the randomized reward
    # is not comparable to the baseline's.
    max_iterations = preset(default=6000, newton_mjwarp=10000)
    experiment_name = "g1_rough_dr"

    def __post_init__(self):
        super().__post_init__()

        # Randomization widens the observation ranges enough that raw inputs no longer sit in a
        # comparable scale across terms; the baseline can skip this, this task cannot. The exported
        # policy carries the normalizer with it.
        self.actor.obs_normalization = True
        self.critic.obs_normalization = True


@configclass
class G1RoughDR29PPORunnerCfg(G1RoughDRPPORunnerCfg):
    """Runner for the rough task on the current robot description.

    Same budget and observation split as :class:`G1RoughDRPPORunnerCfg`; only the experiment name
    differs, so the two assets' runs do not land in one log directory and get compared by accident.
    """

    experiment_name = "g1_rough_dr29"


@configclass
class G1FlatDRPPORunnerCfg(G1FlatPPORunnerCfg):
    """Runner for the flat task with friction/armature/stiffness randomization.

    Identical to the stock flat runner apart from the name, the budget, and the observation split, so
    a comparison against ``g1_flat`` isolates the randomization and the observation window rather
    than a hyperparameter change.
    """

    # The actor gets only the deployable proprioception; the critic additionally gets the base
    # linear velocity the robot cannot measure. Without this line the ``critic`` group defined on
    # G1FlatDRObservationsCfg would simply never be read.
    obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_flat_dr"
        self.max_iterations = 3000
