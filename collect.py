import argparse
import sys
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
import real2simenv
from omni.isaac.lab_tasks.utils import parse_env_cfg

from omegaconf import OmegaConf
from wrappers import DataCollector
from datetime import datetime
from planning.orchestrator import Orchestrator
from scripts.xform_mapper import GUI_matrix_to_pos_and_quat
import yaml
from config import Config

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: Config):
    # Load configuration
    with open(cfg.usd_info_path, "r") as usd_info_file:
        usd_info = yaml.safe_load(usd_info_file)
        cfg.usd_info = usd_info

    # create environment configuration
    env_cfg: real2simenv.Real2SimCfg = parse_env_cfg(
        args_cli.task,
        device= args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.setup(cfg)

    # video wrapper stuff
    env_cfg.viewer.resolution = cfg.video.viewer_resolution
    env_cfg.viewer.eye = cfg.video.viewer_eye
    env_cfg.viewer.lookat = cfg.video.viewer_lookat
    video_kwargs = cfg.video

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, custom_cfg=cfg, render_mode="rgb_array")

    # apply wrappers
    if cfg.video.enabled:          
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = DataCollector(env, cfg.data_collection, save_dir=f"data/{args_cli.ds_name}")

    # Reset environment
    env.reset()


    orchestrator = Orchestrator(env, cfg)
    plan_template = [
        # ("grasp", {"target":"bowl"}),
        ("grasp", {"target": "newcube"}),
    ]

    # Simulate environment
    # with torch.inference_mode():
    while simulation_app.is_running():
        full_plan = orchestrator.generate_plan_from_template(plan_template)

        # ignoring using torch inference mode for now
        last_action = None
        done, trunc = False, False
        for segment in full_plan:
            obs, rew, done, trunc, info = env.step(segment)
            last_action = segment
            if done or trunc:
                print("Done or truncated!")
                break

        if not done and not trunc:
            env.reset()

    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
