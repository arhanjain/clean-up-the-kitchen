import sys
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="test")

parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False, help="Disable fabric and use USD I/O operations.",) 
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# parser.add_argument("--name", type=str, required=True, help="Name of the experiment.")
# parser.add_argument("--checkpoint", type=str, default=None, help="Epoch to load.")
# parser.add_argument("--data", type=str, required=True, help="Where to find the attributes of the data trained on")
# parser.add_argument("--replay", action="store_true", help="Replay the data")


AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args # clear out sys.argv for hydra

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

########################################
import cv2
import torch
import hydra
import gymnasium as gym
import cleanup.real2simenv
import yaml
import robomimic.utils.file_utils as FileUtils
import h5py
import json
import random
import numpy as np

from omni.isaac.lab_tasks.utils import parse_env_cfg
from moviepy.editor import ImageSequenceClip
from omegaconf import OmegaConf
from omni.isaac.lab_tasks.utils import parse_env_cfg
from wrappers import DataCollector
# from models import *
# from train import PCDCore
# from config import Config
from cleanup.config import Config
from pathlib import Path
# from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
import requests
import json_numpy
json_numpy.patch()
import numpy as np

def save_vid(vid_arr, idx):
    Path(f"./data/deploy_sim").mkdir(parents=True, exist_ok=True)
    ImageSequenceClip(vid_arr, fps=30).write_videofile(f"./data/deploy_sim/{idx}.mp4", codec="libx264", fps=10) 
    vid_arr.clear()

@hydra.main(version_base=None, config_path="./cleanup/config/", config_name="config")
def main(cfg: Config):
    # Load configuration

    # create environment configuration
    env_cfg: real2simenv.Real2SimCfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.setup(cfg)

    # video wrapper stuff
    env_cfg.viewer.resolution = tuple(cfg.video.viewer_resolution)
    env_cfg.viewer.eye = cfg.video.viewer_eye
    env_cfg.viewer.lookat = cfg.video.viewer_lookat
    video_kwargs = {
        "video_folder": cfg.video.video_folder,
        "step_trigger": lambda step: step % cfg.video.save_steps == 0,
        "video_length": cfg.video.video_length,
    }

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, custom_cfg=cfg, render_mode="rgb_array")

    # apply wrappers
    if cfg.video.enabled:          
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Reset environment
    env.reset()

    # Temp fix to image rendering, so that it captures RGB correctly before entering.
    obs, info = env.reset()



    # Simulate environment
    vid = []
    episode = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            # obs["policy"]["rgb"] = obs["policy"]["rgb"].permute(2, 0, 1)
            rgb = obs["policy"]["rgb"].cpu().numpy()
            vid.append(rgb)
            # img = cv2.resize(rgb, (1000,1000))
            cv2.imshow("img", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                print("Resetting environment!")
                obs, info = env.reset()
                save_vid(vid, episode)
                episode += 1
                continue

            action = requests.post(
                "http://0.0.0.0:8000/act",
                json={
                    "image": rgb,
                    # "instruction": "put the carrot in the sink",
                    "instruction": "open the microwave",
                    "unnorm_key": "bridge_orig",
                    }
            ).json()
            action = torch.tensor(action[None])
            print(f"Action: {action}")
            obs, rew, done, trunc, info = env.step(action)
            if done or trunc:
                save_vid(vid, episode)
                episode += 1


    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
