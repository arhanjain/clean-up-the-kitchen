import torch
import pickle
import gymnasium as gym

from pathlib import Path
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, data_dir):
        '''
        Custom PyTorch Dataset to load trajectory data from disk
        and cache it in memory for faster access.

        Parameters
        ----------
        data_dir : Path
            Path to the directory containing the trajectory data files.
        '''
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory {data_dir} does not exist")

        with open(self.data_dir/"info.pkl", "rb") as f:
            self.info = pickle.load(f)

        self.data = []
        self._len = 0

        self.load_all()

    def __len__(self):
        return self._len

    def __getitem__(self, idx):

        # TODO: replace this with a gymnasium flatten?
        obs = gym.spaces.flatten(self.info["obs_space"], self.data[idx]["obs"])
        act = gym.spaces.flatten(self.info["action_space"], self.data[idx]["action"])

        obs = torch.tensor(obs, dtype=torch.float32)
        act = torch.tensor(act, dtype=torch.float32)

        return obs, act

        # groups = []
        # for group, group_obs in self.data[idx]["obs"].items():
        #     groups.append(torch.cat(list(group_obs.values()), dim=-1))
        # return torch.cat(groups, dim=-1), self.data[idx]["action"]

    def load_all(self):
        '''
        Load all data files into memory.
        '''
        for file_path in self.data_dir.glob("episode_*.pkl"):
            with file_path.open('rb') as f:
                data = pickle.load(f)
                self._len += len(data)
                self.data += data
    
    def get_info(self):
        return self.info
