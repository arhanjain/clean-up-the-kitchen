
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
import customenv
from omegaconf import OmegaConf

# These are all things I tried to make curobo grasp correctly: 

# def pos_and_quat_from_matrix(transform_mat):
#     pos = transform_mat[:3, -1]
#     quat = quat_from_matrix(transform_mat[:3, :3])
#     return pos, quat

def pos_and_quat_from_matrix(transform_mat):
    transform_mat = np.array(transform_mat)  # Convert the list of lists to a numpy array
    pos = torch.tensor(transform_mat[:3, -1], dtype=torch.float32)
    quat = torch.tensor(transform_mat[:3, :3], dtype=torch.float32)  # Convert to tensor for further processing
    quat_res = quat_from_matrix(quat)
      # Quaternion representing a 180-degree rotation around the x-axis
    quat_180_x = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    # Rotate the quaternion by 180 degrees
    rotated_quat = quaternion_multiply(quat_res, quat_180_x)
    return quat, rotated_quat

def quaternion_multiply(q, r):
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.tensor([w, x, y, z], dtype=torch.float32)
# def quaternion_multiply(q, r):
#     w1, x1, y1, z1 = q
#     w2, x2, y2, z2 = r
#     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#     y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
#     z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
#     return torch.tensor([w, x, y, z], dtype=torch.float32)

# def pos_and_quat_from_matrix(transform_mat):
#     pos = torch.tensor(transform_mat[:3, -1], dtype=torch.float32)
#     quat = torch.tensor(transform_mat[:3, :3], dtype=torch.float32)  # Convert to tensor for further processing
#     quat_res = quat_from_matrix(quat)
    
#     # Quaternion representing a 180-degree rotation around the x-axis
#     quat_180_x = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    
#     # Rotate the quaternion by 180 degrees
#     rotated_quat = quaternion_multiply(quat_res, quat_180_x)
    
#     return pos, rotated_quat

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
    env.reset()
    planner = MotionPlanner(env)
    # simulate environment
    print("begin!")
    while simulation_app.is_running():
        # run everything in inference mode
        joint_pos, joint_vel, joint_names = env.get_joint_info()
        meta_data, rgb, seg, depth = env.get_camera_data()
        # Not always consistent, hit or miss, currently using 25 inference passes from M2T2 config
        cfg = OmegaConf.load("/home/jacob/projects/clean-up-the-kitchen/M2T2/config.yaml")
        data, outputs = load_and_predict(cfg, meta_data, rgb, depth, seg)

        # Visualize through meshcat-viewer
        visualize(cfg, data, outputs)
        # Gives transformation matricies for each object
        grasps = np.array(outputs['grasps'])
        grasp_conf = torch.tensor(outputs['grasp_confidence'], dtype = torch.float32)
        grasp_conf = outputs['grasp_confidence']
        # Currently only support for grasping one type of object in one environment

        if grasps.size > 0 and len(grasps[0]) == len(grasp_conf):
            sorted_indices = sorted(range(len(grasp_conf)), key=lambda i: grasp_conf[i], reverse=True)
            grasps = grasps[0][sorted_indices]
            grasp_conf = [grasp_conf[i] for i in sorted_indices]
        else:
            print("Length of grasps and grasp_confidence do not match.")
        sorted_indices = torch.argsort(grasp_conf, descending = True) 
        grasps = grasps[sorted_indices]
        success = False
        for i in range(grasps[0].shape[0]):
            best_grasp = grasps[0][i]
            pos, quat = pos_and_quat_from_matrix(best_grasp)
            goal = torch.cat([pos, quat], dim=0).unsqueeze(0).repeat(env.num_envs, 1).to(env.unwrapped.device)
            plan, success = planner.plan(joint_pos, joint_vel, joint_names, goal, mode="ee_pose")
            if success:
                # print(f"Grasp {i} succeeded with confidence {grasp_conf[sorted_indices[i]]}")
                # print(f"Grasp {i} succeeded with confidence {grasp_conf[i]}")
                break
        if success:
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
        else:
            print("No successful grasp found")
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

