import gymnasium as gym

from .cube_env import CubeEnvCfg
from .wrapper import TestWrapper

gym.register(
    id="Cube-Test-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": CubeEnvCfg
    },
    disable_env_checker=True
)