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
parser.add_argument("--task", type=str, default="Real2Sim", help="Name of the task.")
parser.add_argument("--run_path", type=str, required=True, help="Path to the run")


AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args # clear out sys.argv for hydra

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

########################################
import torch
import hydra
import gymnasium as gym
# import real2simenv
import robomimic.utils.file_utils as FileUtils
import h5py
import json
import numpy as np

from omni.isaac.lab_tasks.utils import parse_env_cfg
# from models import *
# from train import PCDCore
from pathlib import Path
from tqdm import tqdm
from cleanup.config import Config
import cleanup.real2simenv
from moviepy.editor import ImageSequenceClip

@hydra.main(version_base=None, config_path="../cleanup/config", config_name="config")
def main(cfg: Config):
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
    obs, info = env.reset()
    
    run_config = json.load(open(args_cli.run_path + "/config.json"))

    hdf5  = h5py.File(run_config["train"]["data"], "r")
    data = hdf5["data"]
    meta = json.loads(data.attrs["meta"])

    action_min = np.array(meta["action_min"])
    action_max = np.array(meta["action_max"])
    def unnormalize_action(action):
        # from -1,1 to min, max
        return (action + 1) * (action_max - action_min) / 2 + action_min

    rollouts = {}
    for checkpoint in tqdm(list(Path(args_cli.run_path).rglob("*.pth"))):
        tqdm.write(f"Running checkpoint: {checkpoint}")

        # Load model
        policy, _ = FileUtils.policy_from_checkpoint(
                ckpt_path=checkpoint, 
                device=env.unwrapped.device, 
                verbose=False)

        vid = []
        # Simulate environment
        with torch.inference_mode():
            while simulation_app.is_running():
                obs["policy"]["rgb"] = obs["policy"]["rgb"].permute(2, 0, 1)
                act = policy(obs["policy"])
                act = unnormalize_action(act)
                act = torch.tensor(act, dtype=torch.float32).to(env.unwrapped.device).view(1, -1)
                obs, rew, done, trunc, info = env.step(act)
                vid.append(env.render())
                if done or trunc:
                    break
        rollouts[checkpoint] = np.array(vid)

    env.close()
    
    breakpoint()
    big_video = np.concatenate(list(rollouts.values()), axis=2)
    big_video = list(big_video)
    breakpoint()

    # for checkpoint, vid in rollouts.items():

    ImageSequenceClip(vid, fps=30).write_videofile("data/g60_pick_remastered.mp4", codec="libx264", fps=10)



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
