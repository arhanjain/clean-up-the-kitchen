
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

from grasp_utils import load_and_predict, visualize
# from test import load_and_predict
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
from omni.isaac.lab.utils.math import subtract_frame_transforms, quat_from_matrix
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg

from planner import MotionPlanner
from customenv import TestWrapper
from omegaconf import OmegaConf
from customenv.cube_env import pos_and_quat_from_matrix


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
    planner = MotionPlanner(env)
    # Temp fix to image rendering, so that it captures rgb correctly before entering.
    for _ in range(10):
        action = torch.tensor(env.action_space.sample()).to(env.device)
        env.step(action)
    # simulate environment
    print("begin!")
    while simulation_app.is_running():
        # run everything in inference mode
        joint_pos, joint_vel, joint_names = env.get_joint_info()
        rgb, seg, depth, meta_data = env.get_camera_data()
        # Not always consistent, hit or miss, currently using 25 inference passes from M2T2 config
        cfg = OmegaConf.load("/home/jacob/projects/clean-up-the-kitchen/M2T2/config.yaml")
        data, outputs = load_and_predict(cfg, meta_data, rgb, depth, seg)
        # Visualize through meshcat-viewer
        visualize(cfg, data, outputs)
        # Currently only support for grasping one type of object in one environment
        # So if there are four objects with different num grasps, it will not work.
        if len(outputs['grasps']) > 0:
            grasps = np.array(outputs['grasps'][0])  # Just the first object
            grasp_conf = np.array(outputs['grasp_confidence'][0])
            sorted_indices = np.argsort(grasp_conf)[::-1]  # Get indices sorted in descending order
            grasps = grasps[sorted_indices]

            success = False
            for i in range(grasps.shape[0]):
                best_grasp = torch.tensor(grasps[i], dtype=torch.float32)
                pos, quat = pos_and_quat_from_matrix(best_grasp)
                goal = torch.cat([pos, quat], dim=0).unsqueeze(0).repeat(env.num_envs, 1).to(env.unwrapped.device)
                plan, success = planner.plan(joint_pos, joint_vel, joint_names, goal, mode="ee_pose")
                if success:
                    # Move back slightly before grasping
                    pos_back = pos - torch.tensor([0, 0, 0.05]).to(pos.device)  # Adjust the distance as needed
                    goal_back = torch.cat([pos_back, quat], dim=0).unsqueeze(0).repeat(env.num_envs, 1).to(env.unwrapped.device)
                    plan_back, success_back = planner.plan(joint_pos, joint_vel, joint_names, goal_back, mode="ee_pose")
                    if success_back:
                        plan = plan_back + plan  # Append the plan to move back before grasping
                        break

            if success:
                with torch.inference_mode():
                    if not success:
                        env.reset()
                        continue
                    plan = planner.pad_and_format(plan)
                    for pose in plan:
                        gripper = torch.ones(env.num_envs, 1).to(0)
                        action = torch.cat((pose, gripper), dim=1)
                        env.step(action)
                    # Close the gripper
                    gripper_close = torch.zeros(env.num_envs, 1).to(0)
                    for _ in range(10):  # Close the gripper
                        action = torch.cat((pose, gripper_close), dim=1)
                        env.step(action)
        else:
            print("No successful grasp found")
        #env.reset()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

