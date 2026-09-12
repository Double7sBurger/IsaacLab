# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G1 velocity-tracking task ported from WBC-AGILE (``agile.rl_env.tasks.locomotion.g1``).

The MDP terms, actuator model, commands, curricula and RSL-RL configuration under this package are
copied verbatim from WBC-AGILE v1.3.1 so they can be diffed against upstream; only the import paths
were rewritten.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Velocity-G1-WBC-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:G1LowerVelocityEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VelocityPpoRunnerCfg",
    },
)

gym.register(
    id="Velocity-G1-WBC-History-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_history_env_cfg:G1LowerVelocityHistoryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VelocityPpoRunnerCfg",
    },
)
