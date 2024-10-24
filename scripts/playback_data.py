import argparse
import sys
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="test")
parser.add_argument("--data_path", type=str, default="./data/nvidia_poop/")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args() 
sys.argv = [sys.argv[0]] + hydra_args # clear out sys.argv for hydra

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

########################################

import torch
import hydra 
import json
import gymnasium as gym 
import cleanup.real2simenv 
from omni.isaac.lab_tasks.utils import parse_env_cfg

from cleanup.config import Config
import cleanup.real2simenv as real2simenv
import numpy as np
import random
import h5py

from moviepy.editor import ImageSequenceClip
from pathlib import Path
import argparse 
from tqdm import tqdm

# def main():

@hydra.main(version_base=None, config_path="../cleanup/config", config_name="config")
def main(cfg: Config):
    # create environment configuration
    env_cfg: real2simenv.Real2SimCfg = parse_env_cfg(
        "Real2Sim",
        device= args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    env_cfg.setup(cfg)

    # create environment
    env = gym.make("Real2Sim", cfg=env_cfg, custom_cfg=cfg, render_mode="rgb_array")
    # Reset environment
    env.reset()


    print(f"Opening {args_cli.data_path}...")
    h5 = h5py.File(args_cli.data_path, "r")
    data = h5["data"]

    meta = json.loads(data.attrs["meta"])
    demos = random.choices(list(data.keys()), k=10)
    action_min = np.array(meta["action_min"])
    action_max = np.array(meta["action_max"])
    def unnormalize_action(action):
        # from -1,1 to min, max
        return (action + 1) * (action_max - action_min) / 2 + action_min

    playback_dir = Path(args_cli.data_path).parent / "playback"
    playback_dir.mkdir(exist_ok=True)
    for idx, demo in tqdm(enumerate(demos)):
        vid = []
        with torch.inference_mode():
            for i in range(data[demo]["actions"].shape[0]):
                print(data[demo]["actions"][i])
                action = data[demo]["actions"][i]
                # action = unnormalize_action(action)
                action = torch.tensor(action, dtype=torch.float32)[None]

                obs, _, _ , _ , _ = env.step(action)
                img = obs["policy"]["rgb"].squeeze()
                vid.append(img.cpu().numpy())
            env.reset()

        ImageSequenceClip(vid, fps=30).write_videofile(f"{str(playback_dir)}/{idx}.mp4", codec="libx264", fps=10)





    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

 
