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
import numpy as np
import cv2
import gymnasium as gym 
from omni.isaac.lab_tasks.utils import parse_env_cfg

from omegaconf import OmegaConf
from wrappers import DataCollector
from datetime import datetime
import yaml
from cleanup.config import Config
import cleanup.real2simenv as real2simenv
import omni.isaac.lab.utils.math as math
from droid.droid.controllers.oculus_controller import VRPolicy

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
    env_cfg.viewer.resolution = (640, 480)
    env_cfg.viewer.eye = (-0.19, -0.98, 0.93)
    env_cfg.viewer.lookat = (0.89, 0.45, -0.18)

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, custom_cfg=cfg, render_mode="rgb_array")
    # teleop = GELLOPolicy()
    teleop = VRPolicy(
        right_controller = True,
        max_lin_vel = 1,
        max_rot_vel = 1,
        max_gripper_vel = 1,
        spatial_coeff = 1,
        pos_action_gain = 3 ,
        rot_action_gain = 2,
        gripper_action_gain = 3,
        )
    # teleop = Se3Keyboard(
    #         pos_sensitivity=0.05,
    #         rot_sensitivity=0.1,
    #         )
    # viewport = ViewportCameraController(env, )

    # apply wrappers
    env = DataCollector(env, cfg.data_collection, save_dir=f"data/{args_cli.ds_name}")
    #
    # teleop.add_callback("O", lambda: env.reset(skip_save=True))
    # teleop.add_callback("P", lambda: env.reset())
    # Reset environment
    obs, _ = env.reset()
    teleop.reset_state()
    # teleop.reset()

    # Simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            controller_info = teleop.get_info()
            ee_pos = env.scene["ee_frame"].data.target_pos_source[:, 0]
            ee_quat = env.scene["ee_frame"].data.target_quat_source[:, 0]
            ee_euler = math.euler_xyz_from_quat(ee_quat)
            ee_euler = torch.cat(ee_euler, dim=-1)
            ee_state = torch.cat([ee_pos, ee_euler[None]], dim=-1)
            state = {
                    "robot_state": {
                        "cartesian_position": ee_state.cpu().numpy().squeeze(),
                        "gripper_position": 0.0,
                        }
                    }
            action, controller_action_info = teleop.forward(state, include_info=True)

            # remap gripper from [0,1] to [-1,1] and flip
            action[-1] = - (action[-1] - 0.5) * 2

            print(action)
            action = torch.tensor(action[None]).float()
            obs, rew, done, trunc, info = env.step(action)
            
            rgb = obs["policy"]["rgb"].squeeze().cpu().numpy()
            rgb = cv2.resize(rgb, (1000, 1000))
            cv2.imshow("rgb", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            if done or trunc:
                print("Done or truncated!")

            # Check controller termination
            end_traj = controller_info["success"] or controller_info["failure"]
            if end_traj:
                print(f"robot default state: {env.scene['robot'].data.default_joint_pos}")
                print("End of trajectory")
                env.reset(skip_save=controller_info["failure"])
                teleop.reset_state()


    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
