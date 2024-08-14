import torch
import pickle
import gymnasium as gym

from pathlib import Path

class DataCollector(gym.Wrapper):
    def __init__(self, env, cfg, save_dir):
        super().__init__(env)
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
                "action_space": env.action_space,
                "obs_space": env.observation_space,
                }
        with open(self.save_dir/"info.pkl", "wb") as f:
            pickle.dump(metadata, f)

        self.buffer = []
        self.ep = 0
        self.max_ep = cfg.max_episodes


    def reset(self, *, seed=None, options=None):
        # self.buffer = []
        self.save_buffer()
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, rew, done, trunc, info = self.env.step(action)
        
        entry = {
                    'obs': self.to_numpy(obs),
                    'action': self.to_numpy(action),
                    'reward': self.to_numpy(rew),
                    'done': self.to_numpy(done),
                    'truncated': self.to_numpy(trunc),
                }
        self.buffer.append(entry)

        if trunc or done:
            self.save_buffer()

        return obs, rew, done, trunc, info
    
    def save_buffer(self):
        if len(self.buffer) == 0:
            return
        save_path = self.save_dir / f'episode_{self.ep}.pkl'
        with save_path.open('wb') as f:
            pickle.dump(self.buffer, f)

        self.buffer = []
        self.ep += 1

        if self.ep >= self.max_ep:
            self.env.close()

    @staticmethod
    def to_numpy(d: dict | torch.Tensor):
        if isinstance(d, dict):
            return {k: DataCollector.to_numpy(v) for k, v in d.items()}
        return d.cpu().numpy()






