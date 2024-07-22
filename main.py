
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

import torch
import numpy as np
import gymnasium as gym
import omni.isaac.lab.sim as sim_utils

import omni.isaac.lab_tasks
from omni.isaac.lab_tasks.utils import parse_env_cfg

from omni.isaac.lab.assets.asset_base_cfg import AssetBaseCfg
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg

from planner import MotionPlanner
from customenv import TestWrapper
import customenv


def main():
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = TestWrapper(env)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    obs, info = env.reset()
    camera_data = env.get_camera_data()
    planner = MotionPlanner(env)
    # simulate environment
    print("begin!")
    while simulation_app.is_running():
        # run everything in inference mode
        joint_pos, joint_vel, joint_names = env.get_joint_info()
        # goal = torch.tensor([0.5, -0.5, 0.7, 0.707, 0, 0.707, 0]).repeat(env.num_envs, 1).to(env.unwrapped.device)
        goal = env.command_manager.get_command("object_pose")
        plan, success = planner.plan(joint_pos, joint_vel, joint_names, goal, mode="ee_pose")
        with torch.inference_mode():
            if not success:
                env.reset()
                continue
            plan = planner.pad_and_format(plan)
            for pose in plan:
                gripper = torch.ones(env.num_envs, 1).to(0)
                try:
                    action = torch.cat((pose, gripper), dim=1)
                except:
                    breakpoint()
                    plan = planner.plan(joint_pos, joint_vel, joint_names, goal, mode="ee_pose")
                env.step(action)
                im = env.get_camera_data()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
