import argparse
import sys
import sys
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="test")

parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--ds_name", type=str, required=True, help="Name of the dataset.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args() 
sys.argv = [sys.argv[0]] + hydra_args # clear out sys.argv for hydra

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

########################################

import torch
import time
import hydra 
import gymnasium as gym 
import cleanup.real2simenv 
import time
import hydra 
import gymnasium as gym 
import cleanup.real2simenv 
from omni.isaac.lab_tasks.utils import parse_env_cfg

from omegaconf import OmegaConf
from wrappers import DataCollector
from datetime import datetime
from cleanup.planning.orchestrator import Orchestrator
from cleanup.planning.orchestrator import Orchestrator
import yaml
from cleanup.config import Config
import cleanup.real2simenv as real2simenv
from gymnasium.wrappers import TimeLimit

@hydra.main(version_base=None, config_path="./cleanup/config", config_name="config")
def main(cfg: Config):
    # create environment configuration
    env_cfg: real2simenv.Real2SimCfg = parse_env_cfg(
        args_cli.task,
        device= args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.setup(cfg)
    env_cfg.setup(cfg)

    # video wrapper stuff
    env_cfg.viewer.resolution = cfg.video.viewer_resolution
    env_cfg.viewer.eye = cfg.video.viewer_eye
    env_cfg.viewer.lookat = cfg.video.viewer_lookat
    video_kwargs = cfg.video
    env_cfg.viewer.resolution = cfg.video.viewer_resolution
    env_cfg.viewer.eye = cfg.video.viewer_eye
    env_cfg.viewer.lookat = cfg.video.viewer_lookat
    video_kwargs = cfg.video

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, custom_cfg=cfg, render_mode="rgb_array")

    env = TimeLimit(env, max_episode_steps=cfg.data_collection.max_steps_per_episode)
    # apply wrappers
    if cfg.video.enabled:          
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = DataCollector(env, cfg.data_collection, env_cfg, save_dir=f"data/", ds_name=args_cli.ds_name, env_name=args_cli.task)
    env.reset()

    ee_pos = env.unwrapped.scene["ee_frame"].data.target_pos_source[:, 0]
    print("current ee pos", ee_pos)

    print("Initializing orchestrator")
    orchestrator = Orchestrator(env, cfg)
    print("Initialized orchestrator")

    # Simulate environment
    i = 0
    while simulation_app.is_running():
        if env.is_stopped():
            print("Data collection has reached max episodes. Exiting simulation loop.")
            env.close()
            break  # Exit the loop

        obs, info = env.reset()
        done, trunc = False, False
        while not done and not trunc:
            print("current ee_pose", env.unwrapped.scene["ee_frame"].data.target_pos_source[:, 0])
            for segment in orchestrator.run():
                obs, rew, done, trunc, info = env.step(segment)
                if done or trunc:
                    print("Done or truncated!")
                    break

        print("Episode:", i)
        i += 1




if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
