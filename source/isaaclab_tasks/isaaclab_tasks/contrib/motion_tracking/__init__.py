# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 reference-motion tracking with native environment managers."""

import gymnasium as gym

for representation, env_class, agent_class in (
    ("GenCoord", "G1ModelBasedEnvCfg", "TrackingModelBasedCfg"),
    ("MaxCoord", "G1MaxCoordModelBasedEnvCfg", "MaxCoordModelBasedCfg"),
):
    agents = {"model_based_cfg_entry_point": f"{__name__}.config.g1.agents.model_based_cfg:{agent_class}"}
    if representation == "GenCoord":
        agents.update(
            paper_model_based_cfg_entry_point=f"{__name__}.config.g1.agents.model_based_cfg:PaperTrackingModelBasedCfg",
            paper_no_critic_cfg_entry_point=f"{__name__}.config.g1.agents.model_based_cfg:PaperNoCriticModelBasedCfg",
        )
    gym.register(
        id=f"IsaacContrib-Motion-Tracking-{representation}-G1",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.config.g1.model_based_env_cfg:{env_class}",
            "default_agent": "model_based",
            **agents,
        },
    )

gym.register(
    id="IsaacContrib-Motion-Tracking-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_tracking_env_cfg:G1TrackingEnvCfg",
        "default_agent": "supermimic",
        "supermimic_cfg_entry_point": "isaaclab_rl.supermimic.supermimic_cfg:SuperMimicCfg",
    },
)

for robot in ("AnymalD", "Cassie", "UnitreeGo2", "H1"):
    gym.register(
        id=f"IsaacContrib-Motion-Tracking-{robot}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.robot_tracking_env_cfg:{robot}TrackingEnvCfg",
            "default_agent": "supermimic",
            "supermimic_cfg_entry_point": "isaaclab_rl.supermimic.supermimic_cfg:SuperMimicCfg",
        },
    )
