import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="test")
parser.add_argument(
    "--cpu", action="store_true", default=False, help="Use CPU pipeline."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
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
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from customenv import TestWrapper
from omegaconf import OmegaConf
from m2t2.m2t2 import M2T2

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from planning.orchestrator import Orchestrator
from customenv import TestWrapper
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.sb3 import process_sb3_cfg, Sb3VecEnvWrapper
from datetime import datetime
from pxr import Usd, UsdGeom
from curobo.geom.types import WorldConfig
from xform_mapper import GUI_matrix_to_pos_and_quat
import hydra
import yaml

# To enable gravity and make the bowl fall
def do_nothing(env):
    ee_frame_sensor = env.unwrapped.scene["ee_frame"]
    tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
    tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
    gripper = torch.ones(env.unwrapped.num_envs, 1).to(env.unwrapped.device)
    action = torch.cat([tcp_rest_position, tcp_rest_orientation, gripper], dim=-1)
    for _ in range(10):
        env.step(action)

def build_collision_table(cfg):
    from pxr import Usd, UsdGeom
    import yaml

    def get_object_dims(stage, subpath):
        prim = stage.GetPrimAtPath(subpath)
        if not prim:
            raise ValueError(f"Prim not found at path: {subpath}")
        
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(prim).GetRange()
        
        dims = bbox.GetSize()
        dims_list = [dims[0], dims[2], dims[1]]
        
        return dims_list

    try:
        with open(cfg.usd_info_path) as file:
            data = yaml.safe_load(file)
            usd_path = data['usd_path']
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {cfg.usd_info_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise ValueError(f"Failed to open USD file: {usd_path}")

    collision_table = {"cuboid": {}}

    for name, obj in data['xforms'].items():
        if name == "sink":
            continue  # Skip the sink object
        dims = get_object_dims(stage, obj['subpath'])

        # Fetch the transformation matrix
        prim = stage.GetPrimAtPath(obj['subpath'])
        transform = torch.tensor(prim.GetChildren()[0].GetAttribute("xformOp:transform").Get())
        
        # Get position and quaternion
        pos, quat = GUI_matrix_to_pos_and_quat(transform)

        z_offset = 0.02

        pos[2] += z_offset
        
        # Convert pos and quat to tensors
        pos_tensor = torch.tensor(pos)
        quat_tensor = torch.tensor(quat)

        # Concatenate pos and quat to form pose using torch.cat
        pose_tensor = torch.cat((pos_tensor, quat_tensor))
        pose = pose_tensor.tolist()  # Convert to list for YAML output

        obj_cfg = {
            "dims": dims,
            "pose": pose
        }
        collision_table["cuboid"][name] = obj_cfg

    collision_table_path = "/home/jacob/projects/curobo/src/curobo/content/configs/world/collision_table.yml"
    try:
        with open(collision_table_path, 'w') as file:
            yaml.dump(collision_table, file, default_flow_style=False)
        print(f"Collision table saved successfully at {collision_table_path}")
    except IOError as e:
        print(f"Failed to save collision table: {e}")


def main():
    # Load configuration
    with open("config.yml", "r") as file:
        cfg = OmegaConf.create(yaml.safe_load(file))
        # Attach USD info
        with open(cfg.usd_info_path, "r") as usd_info_file:
            cfg.usd_info = yaml.safe_load(usd_info_file)
    
    # build_collision_table(cfg)
    # create environment configuration
    env_cfg = parse_env_cfg(
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
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    # apply wrappers
    env = TestWrapper(env)
    if cfg.video.enabled:          
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Reset environment
    env.reset()

    # Temp fix to image rendering, so that it captures RGB correctly before entering.
    for _ in range(10):
        action = torch.tensor(env.action_space.sample()).to(env.device)
        env.step(action)
    env.reset()

    orchestrator = Orchestrator(env, cfg)

    # Get the scene info and then plan
    # scene_info = Orchestrator.extract_objects_and_sites_info(cfg.usd_info_path)
    # prompt = Orchestrator.generate_cleanup_tasks(scene_info)
    # plan_template = Orchestrator.get_plan(prompt)
    # print('plan template:', plan_template)
    # plan_template = eval(plan_template)
    # plan_template = Orchestrator.parse_plan_template(plan_template)
    # plan_template = [
    #     ("grasp",
    #         {
    #             "target": "bowl",
    #         },
    #     ),
    #     ("place",
    #         {
    #             "target": "bowl",
    #         }
    #      ),
    # ]
    plan_template = [
        ('grasp', {"target": "blue_cup"}), 
        ('place', {"target": "blue_cup"}), 
        ('grasp', {"target": "bowl"}), 
        ('place', {"target": "bowl"}), 
        ('grasp', {"target": "ketchup"}), 
        ('place', {"target": "ketchup"}), 
        ('grasp', {"target": "paper_cup"}), 
        ('place', {"target": "paper_cup"}), 
        ('grasp', {"target": "big_spoon"}), 
        ('place', {"target": "big_spoon"})
    ]
 
    # Not going through plan_template fully 
    # Simulate environment
    while simulation_app.is_running():
        do_nothing(env)
        full_plan = orchestrator.generate_plan_from_template(plan_template)
        
        for segment in full_plan:
            obs, rew, done, trunc, info = env.step(segment)
            
            # if done:
            #     print("finished")
            #     break
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
