import torch
import numpy as np

from enum import Enum
from abc import abstractmethod
from typing import Generator
from dataclasses import dataclass
from cleanup.planning.grasp import Grasper
from cleanup.planning.motion_planner import MotionPlanner
from curobo.util.usd_helper import UsdHelper
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils import math as math_utils
import json_numpy
import requests
json_numpy.patch()

class ServiceName(Enum):
    GRASPER = "grasper"
    MOTION_PLANNER = "motion_planner"
    OPEN_VLA = "open_vla"

@dataclass(frozen=True)
class Action:
    _registry = {}
    _services = {}

    def __init_subclass__(cls, action_name, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        Action._registry[action_name] = cls

    @classmethod
    def register_service(cls, service_name, service):
        '''
        Register a service for use by actions.
        Parameters
        ----------
        service_name : str
            Name of the service
        service : object
            Service object
        '''
        Action._services[service_name] = service

    @classmethod
    def get_service(cls, service_name):
        '''
        Get a registered service.
        Parameters
        ----------
        service_name : str
            Name of the service
        Returns
        -------
        object
            Service object
        '''
        service = Action._services.get(service_name, None)
        if service is None:
            raise ValueError(f"Service {service_name} not found in registry")
        return service
        
    @classmethod
    def create(cls, action_name, *args, **kwargs):
        '''
        Factory method to create a specific action object.

        Parameters
        ----------
        action_name : str
            Name of the action to create
        *args, **kwargs
            Arguments to pass to the action constructor

        Returns
        -------
        action: Action
            Action object

        '''
        action_cls = Action._registry.get(action_name, None)
        if action_cls is None:
            raise ValueError(f"Action {action_name} not found in registry")
        return action_cls(*args, **kwargs)

    @abstractmethod
    def build(self, env) -> Generator[torch.Tensor, None, None]:
        '''
        Builds a generator for a sequence of robot executable actions.
        N = number of environments
        A = dimension of action space

        Yields
        ------
        torch.Tensor((N, A), dtype=torch.float32)
            Next executable action in action sequence
        '''
        pass

@dataclass(frozen=True)
class GraspAction(Action, action_name="grasp"):
    target: str
    GRASP_STEPS: int = 7

    def build(self, env):
        # Ensure required services are registered
        grasper: Grasper = Action.get_service(ServiceName.GRASPER)
        planner: MotionPlanner = Action.get_service(ServiceName.MOTION_PLANNER)
        if grasper is None:
            raise ValueError("Grasper service not found")
        if planner is None:
            raise ValueError("Motion planner service not found")
        planner.motion_gen.clear_world_cache()
        planner.update()

        # Find successful plan
        traj = None
        while traj is None:
            # Get grasp pose
            success = torch.zeros(env.unwrapped.num_envs)
            grasp_pose = None
            while not torch.all(success):
                grasp_pose, success = grasper.get_grasp(env, self.target, viz=True)

            # Get pregrasp pose
            pregrasp_pose = grasper.get_prepose(grasp_pose, 0.05)
            # Plan motion to pregrasp
            print("pregrasp_pose", pregrasp_pose)
            traj = planner.plan(pregrasp_pose, mode="ee_pose_abs")

        # Go to pregrasp pose
        gripper_action = torch.ones(env.unwrapped.num_envs, traj.shape[1], 1).to(env.unwrapped.device)
        traj = torch.cat((traj, gripper_action), dim=2)
        for pose_idx in range(traj.shape[1]):
            yield traj[:, pose_idx]

        # Go to grasp pose
        opened_gripper = torch.ones(env.unwrapped.num_envs, 1)
        go_to_grasp = torch.cat((grasp_pose, opened_gripper), dim=1).to(env.unwrapped.device)
        for _ in range(self.GRASP_STEPS):
            yield go_to_grasp

        # Close gripper
        closed_gripper = -1 * torch.ones(env.unwrapped.num_envs, 1)
        close_gripper = torch.cat((grasp_pose, closed_gripper), dim=1).to(env.unwrapped.device)
        for _ in range(self.GRASP_STEPS):
            yield close_gripper
        planner.motion_gen.clear_world_cache()
        planner.update()
        planner.attach_obj(self.target)
        
        # Go to pregrasp pose
        go_to_pregrasp = torch.cat((pregrasp_pose, closed_gripper), dim=1).to(env.unwrapped.device)
        for _ in range(self.GRASP_STEPS):
            yield go_to_pregrasp

@dataclass(frozen=True)
class PlaceAction(Action, action_name="place"):
    target: str
    GRASP_STEPS: int = 7

    def build(self, env):
        # Ensure required services are registered
        grasper: Grasper = Action.get_service(ServiceName.GRASPER)
        planner: MotionPlanner = Action.get_service(ServiceName.MOTION_PLANNER)
        if grasper is None:
            raise ValueError("Grasper service not found")
        if planner is None:
            raise ValueError("Motion planner service not found")
        
        # Find successful plan
        traj = None
        while traj is None:
            # Get place pose
            # success = torch.zeros(env.unwrapped.num_envs)
            # place_pose = None
            # while not torch.all(success):
            #     place_pose, success = grasper.get_placement(env, self.target)
            
            place_pose = grasper.get_placement(env, self.target)
            # Get pregrasp pose
            preplace_pose = grasper.get_prepose(place_pose, 0.1)
            # Plan motion to pregrasp
            traj = planner.plan(preplace_pose, mode="ee_pose_abs")
        
        # Go to preplace pose
        gripper_action = -1 * torch.ones(env.unwrapped.num_envs, traj.shape[1], 1).to(env.unwrapped.device)
        traj = torch.cat((traj, gripper_action), dim=2)
        for pose_idx in range(traj.shape[1]):
            yield traj[:, pose_idx]
        # Go to place pose
        closed_gripper = -1 * torch.ones(env.unwrapped.num_envs, 1)
        go_to_grasp = torch.cat((place_pose, closed_gripper), dim=1).to(env.unwrapped.device)
        for _ in range(self.GRASP_STEPS):
            yield go_to_grasp

        # open gripper
        opened_gripper = torch.ones(env.unwrapped.num_envs, 1)
        open_gripper = torch.cat((place_pose, opened_gripper), dim=1).to(env.unwrapped.device)
        for _ in range(self.GRASP_STEPS):
            yield open_gripper

        planner.detach_obj()
        planner.motion_gen.clear_world_cache()
        planner.update()
        
        # Go to pregrasp pose
        go_to_pregrasp = torch.cat((preplace_pose, opened_gripper), dim=1).to(env.unwrapped.device)
        for _ in range(self.GRASP_STEPS):
            yield go_to_pregrasp


@dataclass(frozen=True)
class RolloutAction(Action, action_name="rollout"):
    instruction: str
    horizon: int = 50

    def build(self, env):
        # Ensure required services are registered
        # model, processor = Action.get_service(ServiceName.OPEN_VLA)
        #
        for _ in range(self.horizon):
            rgb, _, _, _ = env.get_camera_data()

            action = requests.post(
                    "http://0.0.0.0:8000/act",
                    json = {
                        "image": rgb.squeeze().astype(np.uint8),
                        "instruction": self.instruction,
                        "unnorm_key": "bridge_orig",
                        }
                    ).json()
            
            # transform gripper action from 0-1 to -1, 1
            action = action.copy()
            gripper = action[-1]
            gripper = 2 * gripper - 1
            action[-1] = gripper
        
            yield torch.tensor(action).unsqueeze(0)


@dataclass(frozen=True)
class OpenDrawerAction(Action, action_name="open_drawer"):
    GRASP_STEPS: int = 20  # Increased for slower movement
    delta_x: float = 0.02  # Smaller delta for slower drawer opening

    def build(self, env):
        # Ensure required services are registered
        grasper: Grasper = Action.get_service(ServiceName.GRASPER)
        planner: MotionPlanner = Action.get_service(ServiceName.MOTION_PLANNER)
        if grasper is None:
            raise ValueError("Grasper service not found")
        if planner is None:
            raise ValueError("Motion planner service not found")

        for _ in range(1):
            yield torch.rand(env.action_space.shape, device=env.unwrapped.device)

        # # planner.update()
        # init_pos = torch.as_tensor([0.5, 0.0, 0.70, 0.5, -0.5, 0.5, -0.5]).to(env.unwrapped.device)
        # handle_id, handle_name = env.scene["kitchen01"].find_bodies("drawer_02_handle")
        # handle_location = env.scene["kitchen01"]._data.body_state_w[0][handle_id][:, :3]
        # offset = torch.tensor([-0.08, 0.00, 0.00]).to(env.unwrapped.device)
        # init_pos[:3] = handle_location + offset

        # grasp_pose = init_pos.unsqueeze(0)
        # print(grasp_pose)

        # # Get pregrasp pose
        # pregrasp_pose = grasper.get_prepose(grasp_pose, 0.1)
        # traj = planner.plan(pregrasp_pose, mode="ee_pose_abs")
        # if traj is None:
        #     raise ValueError("Failed to plan to pregrasp pose")
        
        # # Go to pregrasp pose
        # gripper_action = torch.ones(env.unwrapped.num_envs, traj.shape[1], 1).to(env.unwrapped.device)
        # traj = torch.cat((traj, gripper_action), dim=2)
        # for pose_idx in range(traj.shape[1]):
        #     yield traj[:, pose_idx]
        # print("going to grasp")

        # # Go to grasp pose
        # # opened_gripper = torch.ones(env.unwrapped.num_envs, 1).to(env.unwrapped.device)
        # # go_to_grasp = torch.cat((grasp_pose, opened_gripper), dim=1)
        # for _ in range(self.GRASP_STEPS):
        #     traj[:, -1, 0] += 0.005
        #     yield traj[:, -1]

        # # Close gripper (make the gripper closing slower)
        # closed_gripper = -1 * torch.ones(env.unwrapped.num_envs, 1).to(env.unwrapped.device)
        # close_gripper = torch.cat((grasp_pose, closed_gripper), dim=1)
        # for _ in range(self.GRASP_STEPS - 10):
        #     yield close_gripper

        # # planner.update()
        # # planner.attach_obj('/World/envs/env_0/kitchen02/drawer_02_handle/handle')
        # print("finished grasp")

        # # From the current grasp pose, pull the drawer out slowly 
        # go_backwards = torch.cat((grasp_pose, closed_gripper), dim=1).to(env.unwrapped.device)
        # for _ in range(self.GRASP_STEPS + 30):
        #     go_backwards[:, 0] -= 0.005
        #     yield go_backwards
        
        # Slowly release the gripper
        # go_to_pregrasp[:, -1] = 1
        # for _ in range(self.GRASP_STEPS):
        #     yield go_to_pregrasp

        # planner.update()

        # planner.detach_obj()



@dataclass(frozen=True)
class OpenCabinetAction(Action, action_name="open_cabinet"):
    GRASP_STEPS: int = 20  # Increased for slower movement
    delta_x: float = 0.02  # Smaller delta for slower drawer opening

    def build(self, env):
        # Ensure required services are registered
        grasper: Grasper = Action.get_service(ServiceName.GRASPER)
        planner: MotionPlanner = Action.get_service(ServiceName.MOTION_PLANNER)
        if grasper is None:
            raise ValueError("Grasper service not found")
        if planner is None:
            raise ValueError("Motion planner service not found")

        # planner.motion_gen.clear_world_cache()
        # planner.update()
        
        grasp_pose = grasper.get_open_grasp_pose(env)
        # Get pregrasp pose
        pregrasp_pose = grasper.get_prepose(grasp_pose, 0.1)
        
        traj = planner.plan(pregrasp_pose, mode="ee_pose_abs")
        if traj is None:
            raise ValueError("Failed to plan to pregrasp pose")
        
        # Go to pregrasp pose
        gripper_action = torch.ones(env.unwrapped.num_envs, traj.shape[1], 1).to(env.unwrapped.device)
        traj = torch.cat((traj, gripper_action), dim=2)
        for pose_idx in range(traj.shape[1]):
            yield traj[:, pose_idx]
        print("going to grasp")

        # Go to grasp pose
        opened_gripper = torch.ones(env.unwrapped.num_envs, 1).to(env.unwrapped.device)
        go_to_grasp = torch.cat((grasp_pose, opened_gripper), dim=1)
        for _ in range(self.GRASP_STEPS - 15):
            yield go_to_grasp

        # Close gripper (make the gripper closing slower)
        closed_gripper = -1 * torch.ones(env.unwrapped.num_envs, 1).to(env.unwrapped.device)
        close_gripper = torch.cat((grasp_pose, closed_gripper), dim=1)
        for _ in range(self.GRASP_STEPS - 10):
            yield close_gripper

        # planner.motion_gen.clear_world_cache()
        # planner.update()
        # planner.attach_obj('/World/envs/env_0/kitchen01/drawer_01_handle/handle')
        print("finished grasp")

        # From the current grasp pose, pull the drawer out slowly 
        go_backwards = torch.cat((grasp_pose, closed_gripper), dim=1).to(env.unwrapped.device)
        for _ in range(self.GRASP_STEPS + 25):
            go_backwards[:, 0] -= 0.01
            go_backwards[:, 1] += 0.005
            yield go_backwards
        
        # Slowly release the gripper
        go_backwards[:, -1] = 1
        for _ in range(self.GRASP_STEPS - 15):
            yield go_backwards
        # planner.motion_gen.clear_world_cache()
        # planner.update()

        # planner.detach_obj()

@dataclass(frozen=True)
class VisualizePCDAction(Action, action_name="visualize"):
    
    def build(self, env, cfg):
        rgb, seg, depth, metadata = env.get_camera_data()
        rgb_image = rgb[0]
        depth_image = depth[0]
        intrinsics = metadata[0]["intrinsics"]

        # Generate point cloud and rgb points
        points = env.depth_to_pointcloud(depth_image, intrinsics)
        h, w, _ = rgb_image.shape
        rgb_flat = rgb_image.reshape(-1, 3)
        depth_flat = depth_image.flatten()
        valid = depth_flat > 0
        rgb_points = rgb_flat[valid]

        # Visualize original point cloud
        env.visualize_pointcloud_open3d(points, rgb_points)

        # Save everything to pointcloud_data.npz
        np.savez("pointcloud_data.npz", rgb_image=rgb_image, depth_image=depth_image,
                 points=points, rgb_points=rgb_points, intrinsics=intrinsics)
        # Initialize ObjectSegmenter with configuration parameters

         # from hydra.core.global_hydra import GlobalHydra
        # if GlobalHydra.instance().is_initialized():
        #     GlobalHydra.instance().clear()
        # from cleanup.planning.object_segmentation import ObjectSegmenter
        # breakpoint()
        # segmenter = ObjectSegmenter()
        # text_prompt = "handle."
        # masks, boxes, labels, scores = segmenter.get_masks(rgb_image, text_prompt)
        # if masks is None:
        #     print("No objects detected.")
        # else:
        #     num_samples = 500
        #     upscaled_points_list = segmenter.upscale_points(masks, num_samples=num_samples)

        #     object_points = []
        #     for upscaled_points in upscaled_points_list:
        #         mask_indices = upscaled_points[:, 0] * w + upscaled_points[:, 1]
        #         mask_valid = depth_image.flatten()[mask_indices] > 0
        #         mask_indices = mask_indices[mask_valid]
        #         obj_points = points[mask_indices]
        #         obj_rgb = rgb_points[mask_indices]
        #         object_points.append((obj_points, obj_rgb))

        #     for i, (obj_points, obj_rgb) in enumerate(object_points):
        #         print(f"Visualizing Object {i+1}")
        #         env.visualize_pointcloud_open3d(obj_points, obj_rgb)

        # # Yield a no-op action to keep the robot stationary
        # noop_action = torch.zeros((env.unwrapped.num_envs, env.action_space.shape[0]), dtype=torch.float32).to(env.unwrapped.device)
        # yield noop_action
        