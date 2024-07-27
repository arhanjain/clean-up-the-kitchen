
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
import os
import torch
import numpy as np
import gymnasium as gym
import customenv
from grasp_utils import load_and_predict, visualize, m2t2_grasp_to_pos_and_quat
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from planner import MotionPlanner
from customenv import TestWrapper
from omegaconf import OmegaConf

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from dataclasses import asdict
from customenv import TestWrapper
from planner import MotionPlanner
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.sb3 import process_sb3_cfg, Sb3VecEnvWrapper
from configuration import SB3Cfg, GeneralCfg, VideoCfg
from datetime import datetime

def main():

    # Initialize dataclass configs
    general_cfg = GeneralCfg().to_dict()
    sb_cfg = SB3Cfg().to_dict()
    viewer_cfg = VideoCfg().to_dict()
    log_dir = f"{general_cfg['log_dir']}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.setup(general_cfg)

    #video wrapper stuff
    env_cfg.viewer.resolution = viewer_cfg.pop("viewer_resolution")
    env_cfg.viewer.eye = viewer_cfg.pop("viewer_eye")
    env_cfg.viewer.lookat = viewer_cfg.pop("viewer_lookat")
    video_kwargs = viewer_cfg

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = TestWrapper(env)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # post-process agent config
    agent_cfg = process_sb3_cfg(sb_cfg) 
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps") 

    # wrap for SB3
    env = Sb3VecEnvWrapper(env)
    env.seed(seed=agent_cfg["seed"])

    agent = PPO(policy_arch, env, verbose=1, **agent_cfg)
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

     # callbacks for agent
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=log_dir, name_prefix="model", verbose=2)
    # train the agent
    agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)
    # save the final model
    agent.save(os.path.join(log_dir, "model"))

    # close the simulator
#     # Create environment configuration
#     env_cfg = parse_env_cfg(
#         args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
#     )
#     # Create environment
#     env = gym.make(args_cli.task, cfg=env_cfg)
#     env = TestWrapper(env)

#     # Print info (this is vectorized environment)
#     print(f"[INFO]: Gym observation space: {env.observation_space}")
#     print(f"[INFO]: Gym action space: {env.action_space}")
#     # Reset environment
#     obs, info = env.reset()
#     planner = MotionPlanner(env)
#     # Temp fix to image rendering, so that it captures RGB correctly before entering.
#     for _ in range(10):
#         action = torch.tensor(env.action_space.sample()).to(env.device)
#         env.step(action)

#     # Simulate environment
#     print("begin!")
#     grasp_cfg = OmegaConf.load("./grasp_config.yaml")
#     while simulation_app.is_running():
#         # Run everything in inference mode
#         joint_pos, joint_vel, joint_names = env.get_joint_info()
#         rgb, seg, depth, meta_data = env.get_camera_data()
#         loaded_data = rgb, seg, depth, meta_data

#         # Load and predict grasp points
#         data, outputs = load_and_predict(loaded_data, grasp_cfg)
#         # Visualize through meshcat-viewer, how can we visualize the batches seperatly.  
#         visualize(grasp_cfg, data[0], {k: v[0] for k, v in outputs.items()})

#         # grasp marker
#         marker_cfg = VisualizationMarkersCfg(
#             prim_path="/Visuals/graspviz",
#             markers={
#                 "frame": sim_utils.UsdFileCfg(
#                     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
#                     scale=(0.5, 0.5, 0.5),
#                 ),
#             }
#         )
#         marker = VisualizationMarkers(marker_cfg)

#         # Check if the first env has grasps, hopefully they have grasps TODO improve
#         if len(outputs["grasps"][0]) > 0:
#             backup = None
#             best_grasps = []
#             for i in range(env.num_envs):
#                 # Get grasps per env in highest confidence order
#                 if len(outputs["grasps"][i]) == 0:
#                     # grasps = backup
#                     best_grasps.append(backup)
#                 else:
#                     grasps = np.concatenate(outputs["grasps"][i], axis=0)
#                     grasp_conf = np.concatenate(outputs["grasp_confidence"][i], axis=0)
#                     sorted_grasp_idxs = np.argsort(grasp_conf)[::-1] # high to low confidence
#                     grasps = grasps[sorted_grasp_idxs]
#                     best_grasps.append(grasps[0])
#                     if i == 0:
#                         backup = grasps[0]

#             # Get motion plan to grasp pose
#             best_grasps = torch.tensor(best_grasps)
#             pos, quat = m2t2_grasp_to_pos_and_quat(best_grasps)
#             goal = torch.cat([pos, quat], dim=1)
#             plan, success = planner.plan(joint_pos, joint_vel, joint_names, goal, mode="ee_pose")
            
#             if success:
#                 with torch.inference_mode():
#                     plan = planner.pad_and_format(plan)
                    
#                     # go to grasp pose
#                     for pose in plan:
#                         gripper = torch.ones(env.num_envs, 1).to(0)
#                         action = torch.cat((pose, gripper), dim=1)
#                         env.step(action)
#                         final_pose = plan[-1]

#                         # marker.visualize(final_pose[:, :3], final_pose[:, 3:])
                    
#                     # close gripper
#                     for _ in range(10):
#                         gripper_close = -1 * torch.ones(env.num_envs, 1).to(final_pose.device)
#                         action = torch.cat((final_pose.clone(), gripper_close), dim=1)
#                         env.step(action)
                    
#                     # move gripper to demonstrate "pick"
#                     for _ in range(50):
#                         gripper_close = -1 * torch.ones(env.num_envs, 1).to(final_pose.device)
#                         newpose = torch.cat((final_pose[:, :3] + 0.2*torch.ones(env.num_envs,3).to(0), final_pose[:, 3:]), dim=1)
#                         action = torch.cat((newpose, gripper_close), dim=1)
#                         env.step(action)

#         else:
#             print("No successful grasp found")

#         env.reset() # To avoid inference mode resetting, comment this out.
        
#     # Close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

