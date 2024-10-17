import torch
import pickle
import numpy as np
import gymnasium as gym
from omni.isaac.lab_tasks.utils.data_collector import RobomimicDataCollector
from pathlib import Path

from cleanup.config.config import Config

class DataCollector(gym.Wrapper):
    def __init__(self, env, cfg, env_cfg, save_dir, ds_name, env_name="Real2Sim"):
        super().__init__(env)
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "action_space": env.action_space,
            "obs_space": env.observation_space,
        }
        with open(self.save_dir / "info.pkl", "wb") as f:
            pickle.dump(metadata, f)

        self.ep = 0
        self.max_ep = cfg.max_episodes  # Ensure this is set in your cfg
        self.data_collector = RobomimicDataCollector(
            env_name=env_name,
            directory_path=self.save_dir,
            filename=ds_name, 
            num_demos=self.max_ep,
            flush_freq=1, 
            env_config=env_cfg,
        )

    def reset(self, **kwargs):
        self.data_collector.reset() 
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        print(action)
        obs, rew, done, trunc, info = self.env.step(action)

        self.collect_data(obs, action, rew, done)

        if done or trunc:
            self.data_collector.flush(env_ids=[0])

        return obs, rew, done, trunc, info

    def collect_data(self, obs, action, reward, done):
        obs = obs.get("policy", {})
        self.data_collector.add("obs/ee_pose", obs["ee_pose"])
        self.data_collector.add("obs/gripper_state", obs["gripper_state"])
        self.data_collector.add("obs/joint_state", obs["joint_state"])
        self.data_collector.add("obs/handle_pose", obs["handle_pose"])
        self.data_collector.add("actions", action)
        # Optionally add rewards and dones if needed
        # self.data_collector.add("rewards", np.array([reward]))
        # self.data_collector.add("dones", np.array([done]))

    def close(self):
        self.data_collector.close()
        super().close()

    def is_stopped(self):
        return self.data_collector.is_stopped()
