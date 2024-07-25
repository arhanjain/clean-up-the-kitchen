
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
from grasp_utils import load_and_predict, visualize, pos_and_quat_from_matrix
from omni.isaac.lab_tasks.utils import parse_env_cfg
from planner import MotionPlanner
from customenv import TestWrapper
from omegaconf import OmegaConf


def main():
    # Create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = TestWrapper(env)

    # Print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # Reset environment
    obs, info = env.reset()
    planner = MotionPlanner(env)
    # Temp fix to image rendering, so that it captures RGB correctly before entering.
    for _ in range(10):
        action = torch.tensor(env.action_space.sample()).to(env.device)
        env.step(action)
    # Simulate environment
    print("begin!")
    while simulation_app.is_running():
        # Run everything in inference mode
        joint_pos, joint_vel, joint_names = env.get_joint_info()
        rgb, seg, depth, meta_data = env.get_camera_data()
        # Load and predict grasp points
        cfg = OmegaConf.load("/home/jacob/projects/clean-up-the-kitchen/M2T2/config.yaml")
        data, outputs = load_and_predict(cfg, meta_data, rgb, depth, seg)
        # Visualize through meshcat-viewer, how can we visualize the batches seperatly.  
        visualize(cfg, data, outputs)

        # Check if there are any grasps
        pos = quat = None
        if len(outputs['grasps']) > 0:
            grasps = np.array(outputs['grasps'][0])  # Just the first object
            grasp_conf = np.array(outputs['grasp_confidence'][0])
            sorted_indices = np.argsort(grasp_conf)[::-1]  # Get indices sorted in descending order
            grasps = grasps[sorted_indices]

            success = False
            for i in range(grasps.shape[0]):
                best_grasp = torch.tensor(grasps[i], dtype=torch.float32)
                pos, quat = pos_and_quat_from_matrix(best_grasp)

                # Move directly to the grasp position
                goal = torch.cat([pos, quat], dim=0).unsqueeze(0).repeat(env.num_envs, 1).to(env.unwrapped.device)
                plan, success = planner.plan(joint_pos, joint_vel, joint_names, goal, mode="ee_pose")
                if success:
                    break

            if success:
                with torch.inference_mode():
                    plan = planner.pad_and_format(plan)
                    for pose in plan:
                        gripper = torch.ones(env.num_envs, 1).to(0)
                        action = torch.cat((pose, gripper), dim=1)
                        env.step(action)
                        final_pose = plan[-1]
                        print('final pose', final_pose)
                        for _ in range(10):
                            gripper_close = -1 * torch.ones(env.num_envs, 1).to(final_pose.device)
                            action = torch.cat((final_pose.clone(), gripper_close), dim=1)
                            env.step(action)
        else:
            print("No successful grasp found")

        env.reset() # To avoid inference mode resetting, comment this out.
        
    # Close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

