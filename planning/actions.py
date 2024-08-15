from os import WSTOPSIG
from numpy import who
import torch

from enum import Enum
from abc import abstractmethod
from typing import Generator
from dataclasses import dataclass
from planning.grasp import Grasper
from planning.motion_planner import MotionPlanner

class ServiceName(Enum):
    GRASPER = "grasper"
    MOTION_PLANNER = "motion_planner"

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
    GRASP_STEPS: int = 30

    def build(self, env):
        # Ensure required services are registered
        grasper: Grasper = Action.get_service(ServiceName.GRASPER)
        planner: MotionPlanner = Action.get_service(ServiceName.MOTION_PLANNER)
        if grasper is None:
            raise ValueError("Grasper service not found")
        if planner is None:
            raise ValueError("Motion planner service not found")
        
        planner.list_obstacles()
        # Disable collision for target before grasping
        planner.disable_collision_for_target(self.target) 

        # Find successful plan
        traj = None
        while traj is None:
            # Get grasp pose
            success = torch.zeros(env.unwrapped.num_envs)
            grasp_pose = None
            while not torch.all(success):
                grasp_pose, success = grasper.get_grasp(env, self.target)
            
            # Get pregrasp pose
            pregrasp_pose = grasper.get_prepose(grasp_pose, 0.1)

            # Plan motion to pregrasp
            traj = planner.plan(pregrasp_pose, mode="ee_pose")
        # Go to pregrasp pose
        gripper_action = torch.ones(env.unwrapped.num_envs, traj.shape[1], 1).to(env.unwrapped.device)
        traj = torch.cat((traj, gripper_action), dim=2)
        for pose_idx in range(traj.shape[1]):
            yield traj[:, pose_idx]

        # Calculate distances and attach the closest object to the robot
        ee_pose_at_pregrasp = traj[:, -1, :3].detach().clone()
        planner.attach_closest_object_to_robot(ee_pose_at_pregrasp)

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

        # Go to pregrasp pose
        go_to_pregrasp = torch.cat((pregrasp_pose, closed_gripper), dim=1).to(env.unwrapped.device)
        for _ in range(self.GRASP_STEPS):
            yield go_to_pregrasp

@dataclass(frozen=True)
class PlaceAction(Action, action_name="place"):
    target: str
    GRASP_STEPS: int = 30

    def build(self, env):
        # Ensure required services are registered
        grasper: Grasper = Action.get_service(ServiceName.GRASPER)
        planner: MotionPlanner = Action.get_service(ServiceName.MOTION_PLANNER)
        if grasper is None:
            raise ValueError("Grasper service not found")
        if planner is None:
            raise ValueError("Motion planner service not found")
        
        # Technically unnecesarry since it's already attached to the robot.
        # planner.disable_collision_for_target(self.target)

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
            traj = planner.plan(preplace_pose, mode="ee_pose")
        
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

        # Detach from robot once it has been dropped
        planner.detach_object_from_robot()
        planner.enable_collision_for_target(self.target)
        
        # Go to pregrasp pose
        go_to_pregrasp = torch.cat((preplace_pose, opened_gripper), dim=1).to(env.unwrapped.device)
        for _ in range(self.GRASP_STEPS):
            yield go_to_pregrasp

