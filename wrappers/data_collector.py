import torch
import pickle
import numpy as np
import gymnasium as gym

from pathlib import Path

class DataCollector(gym.Wrapper):
    def __init__(self, env, cfg, save_dir):
        super().__init__(env)
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # temporary way to extract camera intrinsics
        # intrinsics = self.env.scene["camera"].data.intrinsic_matrices[0].cpu().numpy()
    
    
        metadata = {
                "action_space": env.action_space,
                "obs_space": env.observation_space,
                }
        with open(self.save_dir/"info.pkl", "wb") as f:
            pickle.dump(metadata, f)


        self.ep = 0
        self.max_ep = cfg.max_episodes

        self._reset_buffer()
        self._last_obs = None

        self.LEN_THRESHOLD = 25


    def reset(self, *, seed=None, options=None):
        self._save_buffer()
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, rew, done, trunc, info = self.env.step(action)
        for i in range(self.env.num_envs):
            self.buffer[i]["observations"].append(self.to_numpy(self._last_obs, idx=i))
            self.buffer[i]["actions"].append(self.to_numpy(action, idx=i))
            self.buffer[i]["rewards"].append(self.to_numpy(rew, idx=i))
            self.buffer[i]["dones"].append(self.to_numpy(done, idx=i))
            self.buffer[i]["truncateds"].append(self.to_numpy(trunc, idx=i))
            self.buffer[i]["next_observations"].append(self.to_numpy(obs, idx=i))

        self._last_obs = obs

        if trunc or done:
            self._save_buffer()

        return obs, rew, done, trunc, info
    
    def _save_buffer(self):
        if len(self.buffer[0]["observations"]) >= self.LEN_THRESHOLD:
            # iterate through each env instance and save the buffer
            for i in range(self.env.num_envs):
                np.savez(
                        file = self.save_dir / f'episode_{self.ep}.npz', 
                        **self.buffer[i]
                        )
                self.ep += 1

        self._reset_buffer()

        if self.ep >= self.max_ep:
            self.env.close()

    def _reset_buffer(self):
        self.buffer = [{
                "observations": [],
                "actions": [],
                "rewards": [],
                "dones": [],
                "truncateds": [],
                "next_observations": [],
                } for _ in range(self.env.num_envs)] # we have a buffer for each env instance

    @staticmethod
    def to_numpy(d: dict | torch.Tensor, idx=None):
        if isinstance(d, dict):
            return {k: DataCollector.to_numpy(v) for k, v in d.items()}
        return d.cpu().numpy() if idx is None else d.cpu().numpy()[idx]







