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
            collision_cache={"obb": 20, "mesh": 10},
        )

        self.motion_gen = MotionGen(motion_gen_config)
        # print("warming up...")
        # self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False, parallel_finetune=True)
        print("Curobo is ready!")

        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            # enable_graph_attempt=2,
            max_attempts=max_attempts,
            # enable_finetune_trajopt=True,
        )

        # self.usd_help.load_stage(env.scene.stage)
        # self.usd_help.add_world_to_stage(world_cfg_list[0], base_frame="/World")
        # breakpoint()

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
        if result.status == MotionGenStatus.TRAJOPT_FAIL:
            print('TRAJOPT_FAIL')
            return None
        if result.status == MotionGenStatus.IK_FAIL:
            print("IK FAILURE")
            return None

        # Extract the trajectories based on single vs batch
        traj = [result.interpolated_plan] if len(ik_goal) == 1 else result.get_paths()
        for t in traj:
            if t is None:
                return None

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

    def attach_closest_object_to_robot(self, ee_position, offset=[0, 0, 0.01]):
        '''
        Attaches the closest object to the robot's collision model after calculating the distance
        from the grasp pose.
        '''
        obstacle_list = self.list_obstacles()
        closest_object = None
        min_distance = float('inf')

        for obstacle in obstacle_list:
            obstacle_data = self.motion_gen.world_model.get_obstacle(obstacle)

            if obstacle_data is not None:
                obstacle_position = torch.tensor(obstacle_data.pose[:3]).to(ee_position.device)
                distance = torch.norm(ee_position - obstacle_position)
                print(f"{obstacle}: {distance}")

                if distance < min_distance:
                    min_distance = distance
                    closest_object = obstacle

        if closest_object is None:
            raise ValueError("No valid obstacles found.")

        joint_positions, joint_velocities, joint_names = self.env.get_joint_info()

        joint_state = JointState(
            position=joint_positions.clone().detach() * 0.0,
            velocity=joint_velocities.clone().detach() * 0.0,
            acceleration=joint_velocities.clone().detach() * 0.0,
            jerk=joint_velocities.clone().detach() * 0.0,
            joint_names=joint_names
        )

        self.motion_gen.attach_objects_to_robot(
            joint_state,
            [closest_object],
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            world_objects_pose_offset=Pose.from_list([0, 0, 0.01, 1, 0, 0, 0], self.tensor_args),
        )

        print(f"Object {closest_object} attached to robot.")
        return closest_object

    def attach_object_to_robot(self, target_name, offset=[0, 0, 0.01]):
        '''
        Attaches the object to the robot's collision model.
        '''
        joint_positions, joint_velocities, joint_names = self.env.get_joint_info()

        joint_state = JointState(
            position=joint_positions.clone().detach().to(self.tensor_args.device),
            velocity=joint_velocities.clone().detach().to(self.tensor_args.device) * 0.0,
            joint_names=joint_names
        )

        default_quaternion = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.tensor_args.device).unsqueeze(0)

        world_objects_pose_offset = Pose(
            position=torch.tensor([offset], device=self.tensor_args.device),
            quaternion=default_quaternion,
            normalize_rotation=True,
        )

        self.motion_gen.attach_objects_to_robot(
            joint_state,
            [target_name],
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            world_objects_pose_offset=world_objects_pose_offset,
        )
        print(f"Object {target_name} attached to robot.")

    def detach_object_from_robot(self):
        '''
        Detaches any object currently attached to the robot's collision model.
        '''
        self.motion_gen.detach_object_from_robot()
        print("Object detached from robot.")

    def disable_collision_for_target(self, target_name):
        '''
        Disables collision checking for the target object.
        '''
        self.motion_gen.world_collision.enable_obstacle(target_name, False)
        print(f"Collision disabled for target: {target_name}")

    def enable_collision_for_target(self, target_name):
        '''
        Re-enables collision checking for the target object.
        '''
        self.motion_gen.world_collision.enable_obstacle(target_name, True)
        print(f"Collision enabled for target: {target_name}")

    def list_obstacles(self):
        obstacle_list = self.motion_gen.world_collision.get_obstacle_names()
        filtered_list = set()
        for obstacle in obstacle_list:
            # Sink technically should be an obstacle, fix later
            if obstacle is not None and 'sink' not in obstacle:
                print(f"Valid obstacle found: {obstacle}")
                filtered_list.add(obstacle)
        return filtered_list
    

    def update(self, target) -> None:
        ignore_list = [
            "Robot",
            "GroundPlane",
            # f"{target}",
            "X_line",
            "Y_line",
            "Z_line",
            "sphere",
        ]


        obstacles = self.usd_help.get_obstacles_from_stage(
            reference_prim_path="/World/envs/env_0",
            ignore_substring=ignore_list,
        ).get_collision_check_world()

        # for obstacle in obstacles.cuboid:
        #     print("Updating with cuboid obstacle")
        # for obstacle in obstacles.mesh:
        #     print("Updating with mesh obstacle")

        # Update the motion generator's world model
        self.motion_gen.update_world(obstacles)

        self.world_cfg = obstacles

    def update_all_obstacle_poses(self, new_poses: dict):
        """
        Update the poses of all obstacles using CuRobo's built-in update function.
        
        Parameters
        ----------
        new_poses : dict
            A dictionary where keys are obstacle names and values are lists representing new poses.
        """
        for obstacle_name in self.list_obstacles():
            if obstacle_name in new_poses:
                new_pose_list = new_poses[obstacle_name]
                
                if not isinstance(new_pose_list, list) or len(new_pose_list) != 7:
                    raise ValueError(f"Pose for obstacle {obstacle_name} must be a list of 7 elements.")
                
                new_pose = Pose.from_list(new_pose_list, tensor_args=self.tensor_args)

                # Update the obstacle pose in the world model
                self.motion_gen.world_collision.update_obstacle_pose_in_world_model(obstacle_name, new_pose)
                print(f"Updated pose for {obstacle_name}.")
            else:
                print(f"No new pose provided for {obstacle_name}, skipping update.")

    def get_current_poses_of_all_obstacles(self):
        """
        Retrieve the current poses of all obstacles.
        
        Returns
        -------
        dict
            A dictionary where keys are obstacle names and values are Pose objects representing their current poses.
        """
        obstacle_poses = {}
        for obstacle_name in self.list_obstacles():
            try:
                current_pose = self.get_current_pose_of_object(obstacle_name) 
                obstacle_poses[obstacle_name] = current_pose
            except ValueError as e:
                print(e)
        return obstacle_poses

    def get_current_pose_of_object(self, obstacle_name):
        """
        Retrieve the current pose of an obstacle.
        
        Parameters
        ----------
        obstacle_name : str
            The name of the obstacle whose pose is to be retrieved.
        
        Returns
        -------
        Pose
            The current pose of the obstacle.
        """
        obstacle = self.motion_gen.world_model.get_obstacle(obstacle_name)
        if obstacle:
            return obstacle.pose
        else:
            raise ValueError(f"Obstacle {obstacle_name} not found in world collision checker.")
