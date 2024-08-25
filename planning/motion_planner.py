import torch
# CuRobo 
from curobo.geom.sdf.world import CollisionCheckerType 
from curobo.geom.types import WorldConfig, Cuboid, Material, Mesh
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import CudaRobotModelConfig, JointState, RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.sphere_fit import SphereFitType
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    MotionGenStatus,
    PoseCostMetric,
)
from typing import List, Tuple
from pxr import UsdGeom, UsdShade, Gf, Sdf
from curobo.geom.sdf.world_mesh import WorldMeshCollision, WorldCollisionConfig

class MotionPlanner:
    '''
    The motion planning system. Uses the CuRobo library to plan trajectories.

    Parameters
    ----------
    env: ManagerBasedRLEnv
        The environment to plan in.
    '''
    def __init__(self, env):
        self.usd_help = UsdHelper()
        self.usd_help.load_stage(env.scene.stage)
        self.tensor_args = TensorDeviceType()

        offset = 3.0
        pose = Pose.from_list([0,0,0,1,0,0,0])

        for i in range(env.num_envs):
            self.usd_help.add_subroot("/World", f"/World/world_{i}", pose)
            pose.position[0,1] += offset


        robot_cfg = RobotConfig.from_dict(
            load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"], 
            self.tensor_args,
        )

        world_cfg_list = []
        for i in range(env.num_envs):
            world_cfg = WorldConfig.from_dict(
                # load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
                load_yaml("/home/arhan/projects/clean-up-the-kitchen/planning/collision_table.yml") 
            )
            # self.usd_help.add_world_to_stage(world_cfg, base_frame=f"/World/world_{i}")
            # self.world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid)
            world_cfg_list.append(world_cfg)

        print("CUSTOM >>", join_path(get_world_configs_path(), "collision_table.yml"))

        trajopt_dt = None
        optimize_dt = True
        trajopt_tsteps = 16
        trim_steps = None
        max_attempts = 4
        interpolation_dt = 0.1
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg_list,
            self.tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            interpolation_dt=interpolation_dt,
            collision_cache={"obb": 20, "mesh": 30},
        )

        self.motion_gen = MotionGen(motion_gen_config)
        # print("warming up...")
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False, parallel_finetune=True)
        print("Curobo is ready!")

        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            # enable_graph_attempt=2,
            max_attempts=max_attempts,
            # enable_finetune_trajopt=True,
        )

        # self.usd_help.load_stage(env.scene.stage)
        # self.usd_help.add_world_to_stage(world_cfg_list[0], base_frame="/World")

        self.device = env.device

        # Forward kinematics
        self.kin_model = CudaRobotModel(robot_cfg.kinematics)
        self.env = env

    def plan(self, goal, mode="ee_pose") -> torch.Tensor | None:
        '''
        Creates a plan to reach the goal position from the current joint position.
        Supports multiple environments, denoted as N.

        Parameters
        ----------
        goal : torch.Tensor((N, 7))
            Goal position
        mode : str
            Type of plan to return. Options: "joint_pos", "ee_pose"

        Returns
        -------
        traj : torch.Tensor((N, Plan Length, 7), dtype=float32) | None
            List of trajectories for each environment, None if planning failed
        '''
        # update world
        # print("Updating world, reading w.r.t.", "poop")
        # obstacles = self.usd_help.get_obstacles_from_stage()

        # self.update_obstacles() # Update the world with obstacles

        jp, jv, jn = self.env.get_joint_info()
        cu_js = JointState( position=self.tensor_args.to_device(jp),
            velocity=self.tensor_args.to_device(jv) * 0.0,
            acceleration=self.tensor_args.to_device(jv)* 0.0,
            jerk=self.tensor_args.to_device(jv) * 0.0,
            joint_names=jn
        )
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        goal_pos = goal[:, 0:3].clone().detach().to(self.device)
        goal_orientation = goal[:, 3:7].clone().detach().to(self.device)

        # compute curobo sol
        ik_goal = Pose(
            position=goal_pos,
            quaternion=goal_orientation
        )
        self.plan_config.pose_cost_metric = None

        # Differentiate between single and batch planning
        result = None
        # with torch.enable_grad():
        #     self.enable_gradients(cu_js)
        #     self.enable_gradients(ik_goal)
        if len(ik_goal) == 1:
            result = self.motion_gen.plan_single(cu_js, ik_goal[0], self.plan_config)
        else: 
            result = self.motion_gen.plan_batch_env(cu_js, ik_goal, self.plan_config.clone())

        # Check failure cases
        # if result.status == MotionGenStatus.TRAJOPT_FAIL:
        #     print('TRAJOPT_FAIL')
        #     return None
        # if result.status == MotionGenStatus.IK_FAIL:
        #     print("IK FAILURE")
        #     return None
        
        if not torch.all(result.success):
            print("Failed to plan for all environments")
            print(result.status)
            return None

        # Extract the trajectories based on single vs batch
        traj = [result.interpolated_plan] if len(ik_goal) == 1 else result.get_paths()

        # Return in desired format
        if mode == "joint_pos":
            return traj
        elif mode == "ee_pose":
            ee_trajs = [self.kin_model.get_state(t.position) for t in traj]
            ee_trajs = self.format_plan(ee_trajs)
            return ee_trajs 
        else:
            raise ValueError("Invalid mode...")

    @staticmethod
    def enable_gradients(instance):
        for attr_name in dir(instance):
            # Exclude special attributes
            if attr_name.startswith('__'):
                continue
            
            attr_value = getattr(instance, attr_name)
            
            # Check if the attribute is a tensor
            if isinstance(attr_value, torch.Tensor):
                attr_value.requires_grad_(True) 

    @staticmethod
    def format_plan(plan):
        '''
        Formats the plan into the desired format.

        Parameters
        ----------
        plan : list(N)
            List of trajectories for each environment

        Returns
        -------
        torch.Tensor((N, Plan Length, A), dtype=torch.float32)
            Next executable action in action sequence
        '''
        tensors = []
        for t in plan:
            pose = torch.cat((t.ee_position, t.ee_quaternion), dim=1)
            tensors.append(pose)
        return torch.stack(tensors)

    def update(
        self,
        ignore_substring: List[str] = ["Franka", "material", "Plane", "Visuals",],
        robot_prim_path: str = "/World/env_0/robot/panda_link0", # caution when going multienv
    ) -> None:
        print("updating world...")
        obstacles = self.usd_help.get_obstacles_from_stage(
            ignore_substring=ignore_substring, reference_prim_path=robot_prim_path
        ).get_collision_check_world()

        self.motion_gen.update_world(obstacles)
        # self._world_cfg = obstacles

    def attach_obj(
        self,
        obj_name: str,
    ) -> None:

        jp, jv, jn = self.env.get_joint_info()
        cu_js = JointState(
            position=self.tensor_args.to_device(jp),
            velocity=self.tensor_args.to_device(jv) * 0.0,
            acceleration=self.tensor_args.to_device(jv) * 0.0,
            jerk=self.tensor_args.to_device(jv) * 0.0,
            joint_names=jn,
        )

        obstacles = self.motion_gen.world_model
        obj_path = [obs.name for obs in obstacles if obj_name in obs.name]

        assert len(obj_path) > 0, f"Object {obj_name} not found in the world"

        self.motion_gen.attach_objects_to_robot(
            cu_js,
            obj_path,
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            world_objects_pose_offset=Pose.from_list([0, 0, 0.01, 1, 0, 0, 0], self.tensor_args),
        )

    def detach_obj(self) -> None:
        self.motion_gen.detach_object_from_robot()


    
