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
parser.add_argument("--checkpoint", type=str, required=True, help="Epoch to load.")


AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

########################################

import torch
import gymnasium as gym
import real2simenv
import yaml
import robomimic.utils.file_utils as FileUtils
import h5py
import json
import numpy as np

from omni.isaac.lab_tasks.utils import parse_env_cfg
from omegaconf import OmegaConf
from omni.isaac.lab_tasks.utils import parse_env_cfg
from wrappers import DataCollector
# from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg


def do_nothing(env):
    env.step(torch.tensor(env.action_space.sample()).to(env.unwrapped.device))

def main():
    # Load configuration
    with open("config/config.yml", "r") as file:
        cfg = OmegaConf.create(yaml.safe_load(file))
        # Attach USD info
        with open(cfg.usd_info_path, "r") as usd_info_file:
            cfg.usd_info = yaml.safe_load(usd_info_file)

    # create environment configuration
    env_cfg: real2simenv.Real2SimCfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.setup(cfg.usd_info)

    # video wrapper stuff
    viewer_cfg = cfg.video.copy()
    env_cfg.viewer.resolution = viewer_cfg.pop("viewer_resolution")
    env_cfg.viewer.eye = viewer_cfg.pop("viewer_eye")
    env_cfg.viewer.lookat = viewer_cfg.pop("viewer_lookat")
    video_kwargs = viewer_cfg

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

    # Load model
    policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_path=args_cli.checkpoint, 
            device=env.unwrapped.device, 
            verbose=True)
      
    
    # obs_space = gym.spaces.flatten_space(env.observation_space).shape[0]
    # action_space = gym.spaces.flatten_space(env.action_space).shape[0]
    # model = MLP(obs_space, action_space)
    # saved = torch.load(f"./runs/{args_cli.name}/model_{args_cli.checkpoint}.pth")
    # model.load_state_dict(saved["model_state_dict"])
    # model = model.to(env.unwrapped.device)

    hdf5  = h5py.File("./data/ds-2024-08-15_12-36-03/data.hdf5", "r")
    data = hdf5["data"]
    meta = json.loads(data.attrs["meta"])

    action_min = np.array(meta["action_min"])
    action_max = np.array(meta["action_max"])

    def unnormalize_action(action):
        # from -1,1 to min, max
        return (action + 1) * (action_max - action_min) / 2 + action_min


    # Simulate environment
    with torch.inference_mode():
        while simulation_app.is_running():
            # obs = DataCollector.to_numpy(obs)
            # obs = gym.spaces.flatten(env.observation_space, obs)
            # obs = torch.tensor(obs, dtype=torch.float32).to(env.unwrapped.device)
            # TODO WE NEED TO UNNORMALIZE
            act = policy(obs["policy"])
            act = unnormalize_action(act)
            act = torch.tensor(act, dtype=torch.float32).to(env.unwrapped.device).view(1, -1)
            obs, rew, done, trunc, info = env.step(act)

    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
