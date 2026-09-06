# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Rough-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "default_agent": "rsl_rl",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-DR",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_dr_env_cfg:G1RoughDREnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughDRPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-DR-Teacher",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_dr_depth_distill_env_cfg:G1RoughDRTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughDRTeacherPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-DR-DepthDistill",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_dr_depth_distill_env_cfg:G1RoughDRDepthDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughDRDepthDistillationRunnerCfg",
        "default_agent": "rsl_rl",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-DR29-OfficialReward",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_dr29_official_env_cfg:G1RoughDR29OfficialRewardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughDR29PPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-DR29-OfficialTeacher",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_dr29_official_env_cfg:G1RoughDR29OfficialTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughDR29PPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-G1-DR29-Teacher-SelfCol",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_dr29_official_env_cfg:G1RoughDR29TeacherSelfCollisionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughDR29PPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-DR29-Teacher-HardPush",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_dr29_official_env_cfg:G1RoughDR29TeacherHardPushEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughDR29PPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-DR29-Teacher-Robust",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_dr29_official_env_cfg:G1RoughDR29TeacherRobustEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughDR29PPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)



gym.register(
    id="Isaac-Velocity-Rough-G1-DR29-Distill",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_dr29_distill_env_cfg:G1RoughDR29DistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughDR29DistillationRunnerCfg",
        "default_agent": "rsl_rl",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-DR29-DepthDistill",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_dr29_depth_distill_env_cfg:G1RoughDR29DepthDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughDR29DepthDistillationRunnerCfg",
        "default_agent": "rsl_rl",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-DR29-Official",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_dr29_official_env_cfg:G1RoughDR29OfficialEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughDR29PPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-DR29",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_dr29_env_cfg:G1RoughDR29EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughDR29PPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "default_agent": "rsl_rl",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-G1-DR",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_dr_env_cfg:G1FlatDREnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatDRPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-G1-DR29",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_dr29_env_cfg:G1FlatDR29EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatDRPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
