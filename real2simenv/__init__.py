import gymnasium as gym

from .environment import Real2SimRLEnv
from .environment_cfg import Real2SimCfg

gym.register(
    id="Real2Sim",
    entry_point="real2simenv.environment:Real2SimRLEnv",
    kwargs={
        "env_cfg_entry_point": Real2SimCfg
    },
    disable_env_checker=True
)
