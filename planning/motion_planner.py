import torch
# CuRobo 
from curobo.geom.sdf.world import CollisionCheckerType 
from curobo.geom.types import WorldConfig 
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import CudaRobotModelConfig, JointState, RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
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
class MotionPlanner:
    '''
    The motion planning system. Uses the CuRobo library to plan trajectories.

    Parameters
    ----------
    env: ManagerBasedRLEnv
        The environment to plan in.
    '''

    def __init__(self, env):
        usd_help = UsdHelper()

        usd_help.load_stage(env.scene.stage)
        offset = 2.5
        pose = Pose.from_list([0,0,0,1,0,0,0])

        for i in range(env.num_envs):
            usd_help.add_subroot("/World", f"/World/world_{i}", pose)
            pose.position[0,1] += offset

        self.tensor_args = TensorDeviceType()

        robot_cfg = RobotConfig.from_dict(
            load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"], 
            self.tensor_args,
        )
        world_cfg_list = []
        for i in range(env.num_envs):
            world_cfg = WorldConfig.from_dict(
                load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
            )
            # self.add_collision_visuals_to_stage(usd_help, world_cfg, base_frame=f"/World/world_{i}")
            world_cfg_list.append(world_cfg)

        trajopt_dt = None
        optimize_dt = True
        trajopt_tsteps = 16
        trim_steps = None
        max_attempts = 4
        interpolation_dt = 0.01
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg_list,
            self.tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            interpolation_dt=interpolation_dt,
            collision_cache={"obb": 30, "mesh": 100},
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

        usd_help.add_world_to_stage(world_cfg, base_frame="/World") # adds the viz inside
        self.device = env.device

        # Forward kinematics
        self.kin_model = CudaRobotModel(robot_cfg.kinematics)
        self.env = env
        # update obstacles and stuff

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

        # Return in desired format
        if mode == "joint_pos":
            return traj
        elif mode == "ee_pose":
            try:
                ee_trajs = [self.kin_model.get_state(t.position) for t in traj]
            except:
                breakpoint()
            ee_trajs = self.format_plan(ee_trajs)
            return ee_trajs 
        else:
            raise ValueError("Invalid mode...")

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

    def add_collision_visuals_to_stage(self, usd_help, world_cfg, base_frame):
        """
        Adds visual representations of the collision meshes to the USD stage.

        Parameters
        ----------
        usd_help : UsdHelper
            The USD helper instance to modify the USD stage.
        world_cfg : WorldConfig
            The world configuration containing collision objects.
        base_frame : str
            The base USD path where the objects should be added.
        """
        stage = usd_help.stage

        for obj in world_cfg.objects:
            name = obj.name
            dims = obj.dims
            pose = obj.pose

            # Adjust the translation vector: assuming pose[0:3] is in the form [x, y, z]
            pos = pose[:3]
            adjusted_pos = [pos[0], -pos[2], pos[1]]  # Y and Z axis flip to match the USD coordinate system

            # Quaternion for rotation: assuming pose[3:7] is [qw, qx, qy, qz]
            quat = pose[3:7]

            # Create a new Xform (transform) at the specified base frame
            cuboid_path = f"{base_frame}/{name}_collision"
            xform = UsdGeom.Xform.Define(stage, cuboid_path)

            # Build the transformation matrix
            translation = Gf.Matrix4d().SetTranslate(Gf.Vec3d(adjusted_pos))
            rotation = Gf.Matrix4d().SetRotate(Gf.Quatf(quat[0], Gf.Vec3f(quat[1], quat[2], quat[3])))
            scale = Gf.Matrix4d().SetScale(Gf.Vec3d(dims))

            transform_matrix = translation * rotation * scale

            # Apply the transformation matrix
            xform.AddTransformOp().Set(transform_matrix)

            # Create a cube and set its size (1.0 since scaling is handled by the matrix)
            cuboid = UsdGeom.Cube.Define(stage, f"{cuboid_path}/Cube")
            cuboid.GetSizeAttr().Set(1.0)

            # Create a material and shader using UsdShade
            material = UsdShade.Material.Define(stage, f"{cuboid_path}/Material")
            shader = UsdShade.Shader.Define(stage, f"{cuboid_path}/Material/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")

            # Correctly specify the SdfValueTypeName for the color input
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(1.0, 0.0, 0.0))  # Red color
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

            # Bind the material to the cuboid
            UsdShade.MaterialBindingAPI(cuboid.GetPrim()).Bind(material)

            print(f"Added visual cuboid for {name} at {cuboid_path}")



