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
from omni.isaac.lab_tasks.utils import parse_env_cfg

from omegaconf import OmegaConf
from wrappers import DataCollector
from datetime import datetime
from cleanup.planning.orchestrator import Orchestrator
import yaml
from cleanup.config import Config
import cleanup.real2simenv as real2simenv

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
    env = DataCollector(env, cfg.data_collection, save_dir=f"data/{cfg.data_collection.ds_name}")

    # Reset environment
    env.reset()
    time.sleep(0.1)
    env.reset()

    orchestrator = Orchestrator(env, cfg)
    plan_template = [
            # ("reach", {"target": "carrot"}),
            ("rollout", {"instruction": "pick up the carrot", "horizon": 60}),
            # ("replay", {"filepath": "./data/pick_carrot/episode_0_rel.npz"}),
    ]

    # Simulate environment
    # with torch.inference_mode():
    while simulation_app.is_running():
        with torch.inference_mode():
            full_plan = orchestrator.generate_plan_from_template(plan_template)
            done, trunc = False, False
            for segment in full_plan:
                print(segment)
                obs, rew, done, trunc, info = env.step(segment)
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
