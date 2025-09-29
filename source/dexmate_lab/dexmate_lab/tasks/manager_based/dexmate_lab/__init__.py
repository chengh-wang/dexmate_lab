# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .dexmate_lab_env_cfg import DexmateLabEnvCfg
from .dexmate_lab_env_cfg_ik import DexmateLabEnvCfgIK
##
# Register Gym environments.
##

# gym.register(
#     id="Isaac-Grasp-Cube-Vega-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": DexmateLabEnvCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.VegaRoughPPORunnerCfg,
#     },
#     disable_env_checker=True,
# )

# Teleoperation environment (IK control)
gym.register(
    id="Isaac-Grasp-Cube-Vega-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": DexmateLabEnvCfgIK,
    },
    disable_env_checker=True,
)

gym.register(
    id="Template-Dexmate-Lab-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexmate_lab_env_cfg:DexmateLabEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)