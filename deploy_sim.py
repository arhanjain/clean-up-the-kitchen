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
parser.add_argument("--checkpoint", type=str, default=None, help="Epoch to load.")
parser.add_argument("--data", type=str, required=True, help="Where to find the attributes of the data trained on")
parser.add_argument("--replay", action="store_true", help="Replay the data")


AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args # clear out sys.argv for hydra

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

########################################
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
from omegaconf import OmegaConf
from omni.isaac.lab_tasks.utils import parse_env_cfg
from wrappers import DataCollector
from models import *
from train import PCDCore
# from config import Config
from cleanup.config import Config
# from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg


def do_nothing(env):
    env.step(torch.tensor(env.action_space.sample()).to(env.unwrapped.device))

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
    do_nothing(env)
    obs, info = env.reset()


    hdf5  = h5py.File(args_cli.data, "r")
    data = hdf5["data"]
    meta = json.loads(data.attrs["meta"])

    action_min = np.array(meta["action_min"])
    action_max = np.array(meta["action_max"])
    def unnormalize_action(action):
        # from -1,1 to min, max
        return (action + 1) * (action_max - action_min) / 2 + action_min

    if args_cli.replay:
        print("Replaying data...")
        demo = random.choice(list(data.keys()))
        data = data[demo]
        # obs = data["states"]
        actions = data["actions"]
        for i in range(actions.shape[0]):
            act = unnormalize_action(actions[i])
            # act = actions[i]
            print(f"Action: {act}")
            act = torch.tensor(act, dtype=torch.float32).to(env.unwrapped.device).view(1, -1)
            # prev_pos = obs["policy"]["ee_pose"][:, :3]
            obs, rew, done, trunc, info = env.step(act)
            # pos = obs["policy"]["ee_pose"][:, :3]

            # diff = pos-prev_pos
            # if torch.norm(diff - act[:, :3]) > 1e-3:
            #     print(f"Error: {torch.norm(diff - act[:, :3])}")
            #     breakpoint()
            # print(f"Obs: {obs['policy']['ee_pose']}")
        env.close()
    else:
        # Load model
        policy, _ = FileUtils.policy_from_checkpoint(
                ckpt_path=args_cli.checkpoint, 
                device=env.unwrapped.device, 
                verbose=True)

        # Simulate environment
        with torch.inference_mode():
            while simulation_app.is_running():
                obs["policy"]["rgb"] = obs["policy"]["rgb"].permute(2, 0, 1)
                act = policy(obs["policy"])
                act = unnormalize_action(act)
                act = torch.tensor(act, dtype=torch.float32).to(env.unwrapped.device).view(1, -1)
                print(f"Action: {act}")
                obs, rew, done, trunc, info = env.step(act)

        env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
