import gymnasium as gym

from . import cube_env

gym.register(
    id="Cube-Test-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": cube_env.CubeEnvCfg
    },
    disable_env_checker=True
)