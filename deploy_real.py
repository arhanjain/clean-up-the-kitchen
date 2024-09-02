import hydra
import h5py
import json
import random
import numpy as np
from config import Config

from droid.droid.robot_env import RobotEnv

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: Config):
        
    # checks
    assert cfg.deploy.dataset != "", "Please provide a dataset"
    assert cfg.deploy.replay or cfg.deploy.checkpoint != "", "Please provide a checkpoint or set replay to True"

    env = RobotEnv()
    env.reset(randomize=False)

    hdf5  = h5py.File(cfg.deploy.dataset, "r")
    data = hdf5["data"]
    meta = json.loads(data.attrs["meta"])

    action_min = np.array(meta["action_min"])
    action_max = np.array(meta["action_max"])
    def unnormalize_action(action):
        # from -1,1 to min, max
        return (action + 1) * (action_max - action_min) / 2 + action_min
    
    if cfg.deploy.replay:
        demo = random.choice(list(data.keys()))
        data = data[demo]
        actions = data["actions"]
        print(f"Replaying data from {demo}")
        for i in range(actions.shape[0]):
            act = unnormalize_action(actions[i])
            print(f"Action: {act}")
            env.step(act)

        pass
    else:




if __name__ == "__main__":
    main()

