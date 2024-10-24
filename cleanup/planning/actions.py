from curobo.wrap.reacher.motion_gen import MotionGenStatus
import torch
import numpy as np

from enum import Enum
from abc import abstractmethod
from typing import Generator
from dataclasses import dataclass
from cleanup.planning.grasp import Grasper
from cleanup.planning.motion_planner import MotionPlanner
import cleanup.planning.utils as planning_utils
from curobo.util.usd_helper import UsdHelper
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg 
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
import omni.isaac.lab.utils.math as math
import json_numpy
import requests
from collections import deque
from typing import List
from dataclasses import field
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

        planner.update()

        # Find successful plan
        traj = None
        while traj is None:
            # Get grasp pose
            success = torch.zeros(env.unwrapped.num_envs)
            grasp_pose = None
            while not torch.all(success):
                grasp_pose, success = grasper.get_grasp(env, self.target, viz=True)
            
            grasp_pose = grasp_pose.to(env.unwrapped.device)
            # Get pregrasp pose
            pregrasp_pose = grasper.get_prepose(grasp_pose, 0.1).to(env.unwrapped.device)
            # Plan motion to pregrasp
            traj = planner.plan(pregrasp_pose, mode="ee_pose_abs")
            if traj == MotionGenStatus.INVALID_START_STATE_JOINT_LIMITS:
                return 
            elif type(traj) == MotionGenStatus:
                traj = None

        # Go to pregrasp pose
        gripper_action = torch.ones(env.unwrapped.num_envs, 1).to(env.unwrapped.device)
        for pose_idx in range(traj.shape[1]):
            # each pose in traj is our goal pose at each step, turn it into a relative action
            desired_pose = traj[:, pose_idx]
            rel_pos, rel_euler = planning_utils.relative_action_to_target(env, desired_pose)
            rel_action = torch.cat((rel_pos, rel_euler, gripper_action), dim=-1)
            yield rel_action

        # Go to grasp pose
        opened_gripper = torch.ones(env.unwrapped.num_envs, 1).to(env.unwrapped.device)
        # go_to_grasp = torch.cat((grasp_pose, opened_gripper), dim=1).to(env.unwrapped.device)
        for _ in range(self.GRASP_STEPS):
            rel_pos, rel_euler = planning_utils.relative_action_to_target(env, grasp_pose)
            go_to_grasp = torch.cat((rel_pos, rel_euler, opened_gripper), dim=-1)
            yield go_to_grasp

        # Close gripper
        closed_gripper = -1 * torch.ones(env.unwrapped.num_envs, 1).to(env.unwrapped.device)
        # close_gripper = torch.cat((grasp_pose, closed_gripper), dim=1).to(env.unwrapped.device)
        for _ in range(self.GRASP_STEPS):
            rel_pos, rel_euler = planning_utils.relative_action_to_target(env, grasp_pose)
            close_gripper = torch.cat((rel_pos, rel_euler, closed_gripper), dim=-1)
            yield close_gripper

        planner.attach_obj(self.target)
        planner.update()
        
        # Go to pregrasp pose
        # go_to_pregrasp = torch.cat((pregrasp_pose, closed_gripper), dim=1).to(env.unwrapped.device)
        for _ in range(self.GRASP_STEPS):
            rel_pos, rel_euler = planning_utils.relative_action_to_target(env, pregrasp_pose)
            go_to_pregrasp = torch.cat((rel_pos, rel_euler, closed_gripper), dim=-1)
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
            
            place_pose = grasper.get_placement(env, self.target).to(env.unwrapped.device)
            # Get pregrasp pose
            preplace_pose = grasper.get_prepose(place_pose, 0.1).to(env.unwrapped.device)
            # Plan motion to pregrasp
            traj = planner.plan(preplace_pose, mode="ee_pose_abs")
            if traj == MotionGenStatus.INVALID_START_STATE_JOINT_LIMITS:
                return 
            elif type(traj) == MotionGenStatus:
                traj = None

        # Go to pregrasp pose
        closed_gripper = -1 * torch.ones(env.unwrapped.num_envs, 1).to(env.unwrapped.device)
        for pose_idx in range(traj.shape[1]):
            # each pose in traj is our goal pose at each step, turn it into a relative action
            desired_pose = traj[:, pose_idx]
            rel_pos, rel_euler = planning_utils.relative_action_to_target(env, desired_pose)
            rel_action = torch.cat((rel_pos, rel_euler, closed_gripper), dim=-1)
            yield rel_action

        # Go to place pose
        # go_to_grasp = torch.cat((grasp_pose, opened_gripper), dim=1).to(env.unwrapped.device)
        for _ in range(self.GRASP_STEPS):
            rel_pos, rel_euler = planning_utils.relative_action_to_target(env, place_pose)
            go_to_grasp = torch.cat((rel_pos, rel_euler, closed_gripper), dim=-1)
            yield go_to_grasp

        # drop object
        opened_gripper = torch.ones(env.unwrapped.num_envs, 1).to(env.unwrapped.device)
        for _ in range(self.GRASP_STEPS):
            rel_pos, rel_euler = planning_utils.relative_action_to_target(env, place_pose)
            close_gripper = torch.cat((rel_pos, rel_euler, opened_gripper), dim=-1)
            yield close_gripper

        planner.detach_obj()
        planner.update()
        
        # Go to pregrasp pose
        # go_to_pregrasp = torch.cat((pregrasp_pose, closed_gripper), dim=1).to(env.unwrapped.device)
        for _ in range(self.GRASP_STEPS):
            rel_pos, rel_euler = planning_utils.relative_action_to_target(env, preplace_pose)
            go_to_pregrasp = torch.cat((rel_pos, rel_euler, opened_gripper), dim=-1)
            yield go_to_pregrasp
        


@dataclass(frozen=True)
class RolloutRobomimicAction(Action, action_name="rollout_robomimic"):
    # instruction: str
    horizon: int = 50
    keys: List[str] = field(default_factory=lambda: ["rgb"])

    def build(self, env):
        # Ensure required services are registered
        # model, processor = Action.get_service(ServiceName.OPEN_VLA)
        #
        last_obs = None
        obs_buffer = deque(maxlen=1)
        for _ in range(self.horizon):
            obs = env.last_obs["policy"]
            obs_buffer.append(obs)
            if len(obs_buffer) < 2:
                obs_buffer.append(obs)
            obs_horizon = {}
            for key in self.keys:
                if key == "rgb":

                    obs_horizon[key] = np.stack([o[key].cpu().numpy().squeeze().transpose(2,0,1) for o in obs_buffer])
                else:
                    obs_horizon[key] = np.stack([o[key].cpu().numpy() for o in obs_buffer])
            action = requests.post(
                    "http://0.0.0.0:8000/act",
                    json = {
                        # "image": obs_horizon,
                        "input": obs_horizon,
                        # "instruction": self.instruction,
                        }
                    ).json()
 
            # transform gripper action from 0-1 to -1, 1
            # action = action.copy()
            # gripper = action[-1]
            # gripper = 2 * gripper - 1
            # action[-1] = gripper
            # action[-1] = 1        
            yield torch.tensor(action).unsqueeze(0)


@dataclass(frozen=True)
class ReachAction(Action, action_name="reach"):
    target: str

    def build(self, env):
        planner: MotionPlanner = Action.get_service(ServiceName.MOTION_PLANNER)
        # grasper: Grasper = Action.get_service(ServiceName.GRASPER)
        if planner is None:
            raise ValueError("Motion planner service not found")
        # if grasper is None: 
        #     raise ValueError("Grasper service not found")
        
        planner.update()

        # Find successful plan
        traj = None
        while traj is None:
            # Get grasp pose
            success = torch.zeros(env.unwrapped.num_envs)
            grasp_pos, grasp_quat = env.get_object_pose(self.target)
            # grasp_pos = torch.tensor([[0.383,  0.0094,  0.4203]]).cuda()
            grasp_pose = torch.cat((grasp_pos, torch.tensor([[-0.0661,  0.9559, -0.0315,  0.2843]]).cuda()), dim=-1)
            # Get pregrasp pose
            pregrasp_pose = Grasper.get_prepose(grasp_pose, 0.2)
            print(pregrasp_pose)
            # Plan motion to pregrasp
            traj = planner.plan(pregrasp_pose, mode="ee_pose_rel")

        # Go to pregrasp pose
        gripper_action = torch.ones(env.unwrapped.num_envs, traj.shape[1], 1).to(env.unwrapped.device)
        traj = torch.cat((traj, gripper_action), dim=2)
        # for pose_idx in range(traj.shape[1]):



@dataclass(frozen=True)
class ReplayAction(Action, action_name="replay"):
    filepath: str

    def build(self, env):
        traj = dict(np.load(self.filepath, allow_pickle=True))
        for step in range(traj["observations"].shape[0]):
            act = traj["actions"][step][None]
            yield torch.tensor(act)




