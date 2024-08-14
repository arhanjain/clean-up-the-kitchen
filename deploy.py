import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="test")
parser.add_argument(
    "--cpu", action="store_true", default=False, help="Use CPU pipeline."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False, help="Disable fabric and use USD I/O operations.",) 
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--name", type=str, required=True, help="Name of the experiment.")
parser.add_argument("--checkpoint", type=str, required=True, help="Epoch to load.")


AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

########################################

import torch
import gymnasium as gym
import real2simenv
from omni.isaac.lab_tasks.utils import parse_env_cfg
# from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg

from omegaconf import OmegaConf

# from planning.orchestrator import Orchestrator # requires recompiling m2t2 stuff with newer torch
from omni.isaac.lab_tasks.utils import parse_env_cfg
from wrappers.logger import DataCollector
import yaml
from models.GMM import MLP

def do_nothing(env):
    env.step(torch.tensor(env.action_space.sample()).to(env.unwrapped.device))

def main():
    # Load configuration
    with open("config.yml", "r") as file:
        cfg = OmegaConf.create(yaml.safe_load(file))
        # Attach USD info
        with open(cfg.usd_info_path, "r") as usd_info_file:
            cfg.usd_info = yaml.safe_load(usd_info_file)

    # create environment configuration
    env_cfg: real2simenv.Real2SimCfg = parse_env_cfg(
        args_cli.task,
        use_gpu=not args_cli.cpu,
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
    obs_space = gym.spaces.flatten_space(env.observation_space).shape[0]
    action_space = gym.spaces.flatten_space(env.action_space).shape[0]
    model = MLP(obs_space, action_space)
    saved = torch.load(f"./runs/{args_cli.name}/model_{args_cli.checkpoint}.pth")
    model.load_state_dict(saved["model_state_dict"])
    model = model.to(env.unwrapped.device)

    # Simulate environment
    with torch.inference_mode():
        while simulation_app.is_running():
            obs = DataCollector.to_numpy(obs)
            obs = gym.spaces.flatten(env.observation_space, obs)
            obs = torch.tensor(obs, dtype=torch.float32).to(env.unwrapped.device)
            act = model(obs)
            obs, rew, done, trunc, info = env.step(act.unsqueeze(0))

    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
