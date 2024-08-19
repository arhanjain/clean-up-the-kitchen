# Converting dataset into hdf5
import h5py
import time
import json
import pickle
import gymnasium as gym
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")

args = parser.parse_args()

data_dir = Path(args.data_dir) 
hf = h5py.File(data_dir/"data.hdf5", "w")

data = hf.create_group("data")

eps = 0
env_name = None
total_steps = 0
with open(data_dir/"info.pkl", "rb") as f:
    info = pickle.load(f)

action_shape = gym.spaces.flatten_space(info["action_space"]).shape
action_min = np.inf * np.ones(action_shape)
action_max = -np.inf * np.ones(action_shape)

# TODO: action mean and std for Normal distribution normalization
#action_me


for ep_path in tqdm(list(data_dir.glob("episode_*.npz"))):
    start = time.time()

    ep = np.load(ep_path, allow_pickle=True)
    
    observations = ep["observations"]
    actions = ep["actions"]
    states = np.array([gym.spaces.flatten(info["obs_space"], o) for o in observations])
    rewards = ep["rewards"]
    dones = ep["dones"]
    truncates = ep["truncateds"]
    next_observations = ep["next_observations"]

    num_samples = len(observations)

    demo_grp = data.create_group(f"demo_{eps}")
    eps += 1
    total_steps += num_samples

    # action processing
    action_min = np.minimum(action_min, np.min(actions, axis=0))
    action_max = np.maximum(action_max, np.max(actions, axis=0))

    
    demo_grp.attrs["num_samples"] = num_samples
    # demo_grp.create_dataset("observations", data=np.array(observations))
    demo_grp.create_dataset("actions_raw", data = actions)
    demo_grp.create_dataset("states", data=states)
    demo_grp.create_dataset("rewards", data=rewards)
    demo_grp.create_dataset("dones", data=dones)
    demo_grp.create_dataset("truncateds", data=truncates)

    obs_group = demo_grp.create_group("obs")
    next_obs_group = demo_grp.create_group("next_obs")
    for key in observations[0]["policy"]:
        obs_group.create_dataset(key, data=np.array([obs["policy"][key] for obs in observations]))
        next_obs_group.create_dataset(key, data=[obs["policy"][key] for obs in next_observations])

    end = time.time()

    # print(f"Took {end-start} seconds for {ep_path}")

# add normalize data between -1, 1
for ep in tqdm(data):
    demo = data[ep]
    demo.create_dataset("actions", data = 2 * ((demo["actions_raw"] - action_min) / (action_max - action_min)) - 1)



env_args = {
    "env_name": "Real2Sim",
    "env_kwargs": {},
    "type": 2,
}
data.attrs["total"] = total_steps
data.attrs["env_args"] = json.dumps(env_args)

ds_meta = {
        "action_min": action_min.tolist(),
        "action_max": action_max.tolist()
        }
data.attrs["meta"] = json.dumps(ds_meta)

hf.close()
