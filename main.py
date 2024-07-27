
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="test")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

########################################
import os
import torch
import numpy as np
import gymnasium as gym
import omni.isaac.lab_tasks
import customenv

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from dataclasses import asdict
from customenv import TestWrapper
from planner import MotionPlanner
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.sb3 import process_sb3_cfg, Sb3VecEnvWrapper
from configuration import SB3Cfg, GeneralCfg, VideoCfg
from datetime import datetime

def main():
    # Initialize dataclass configs
    general_cfg = GeneralCfg().to_dict()
    sb_cfg = SB3Cfg().to_dict()
    viewer_cfg = VideoCfg().to_dict()
    log_dir = f"{general_cfg['log_dir']}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.setup(general_cfg)

    #video wrapper stuff
    env_cfg.viewer.resolution = viewer_cfg.pop("viewer_resolution")
    env_cfg.viewer.eye = viewer_cfg.pop("viewer_eye")
    env_cfg.viewer.lookat = viewer_cfg.pop("viewer_lookat")
    video_kwargs = viewer_cfg

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = TestWrapper(env)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # post-process agent config
    agent_cfg = process_sb3_cfg(sb_cfg) 
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps") 

    # wrap for SB3
    env = Sb3VecEnvWrapper(env)
    env.seed(seed=agent_cfg["seed"])

    agent = PPO(policy_arch, env, verbose=1, **agent_cfg)
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

     # callbacks for agent
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=log_dir, name_prefix="model", verbose=2)
    # train the agent
    agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)
    # save the final model
    agent.save(os.path.join(log_dir, "model"))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
