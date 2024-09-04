import torch
import numpy as np

from enum import Enum
from abc import abstractmethod
from typing import Generator
from dataclasses import dataclass
from planning.grasp import Grasper
from planning.motion_planner import MotionPlanner
from curobo.util.usd_helper import UsdHelper
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
import omni.isaac.lab.utils.math as math
from transformers import AutoModelForVision2Seq, AutoProcessor

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
            # success = torch.zeros(env.unwrapped.num_envs)
            # grasp_pose = None
            # while not torch.all(success):
            #     grasp_pose, success = grasper.get_grasp(env, self.target)


            # Get pregrasp pose
            # pregrasp_pose = grasper.get_prepose(grasp_pose, 0.1)
            grasp_pose = torch.tensor([[ 0.3619,  0.3573,  0.4376,  0.0166, -0.7524, -0.6581, -0.0240]])
            pregrasp_pose = torch.tensor([[ 0.3619,  0.3573,  0.4376,  0.0166, -0.7524, -0.6581, -0.0240]])
            # Plan motion to pregrasp
            traj = planner.plan(pregrasp_pose, mode="ee_pose")


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

        planner.update()
        planner.attach_obj(self.target)
        
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

        planner.detach_obj()
        planner.update()
        
        # Go to pregrasp pose
        go_to_pregrasp = torch.cat((preplace_pose, opened_gripper), dim=1).to(env.unwrapped.device)
        for _ in range(self.GRASP_STEPS):
            yield go_to_pregrasp

    @dataclass(frozen=True)
    class RollOutAction(Action, action_name="rollout"):
        target: str
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

            ######################## OpenVLA zero-shot ##########################################
            processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
            vla = AutoModelForVision2Seq.from_pretrained(
                "openvla/openvla-7b", 
                attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True, 
                trust_remote_code=True
            ).to("cuda:0")
            
            INSTRUCTION = f'pick up the {self.target}'
            prompt = f'In: What action should the robot take to {INSTRUCTION}?\nOut:'
            rgb, _, _, _ = env.get_camera_data()

            # Check shapes later
            inputs = processor(prompt, rgb).to("cuda:0", dtype=torch.bfloat16)

            # Check what actions it returns, shape, etc
            # Unnormalized (continuous) action vector --> end-effector deltas.
            action_deltas = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

            ee_deltas = action_deltas[:, :-1]  # End-effector pose deltas
            gripper_deltas = action_deltas[:, -1:]  # Gripper deltas

            # This should get absolute end effector, might need to double check though
            next_ee_pose = current_ee_pose + ee_deltas

            next_pose_with_gripper = torch.cat((next_ee_pose, gripper_deltas), dim=1).to(env.unwrapped.device)

            # Might have to return individually, double check
            yield next_pose_with_gripper