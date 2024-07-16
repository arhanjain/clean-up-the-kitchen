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
    PoseCostMetric,
)

    
class MotionPlanner:
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
            usd_help.add_world_to_stage(world_cfg, base_frame=f"/World/world_{i}")
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
            # maximum_trajectory_dt=0.25, 
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

        # usd_help.add_world_to_stage(world_cfg, base_frame="/World")

        self.device = env.device

        # Forward kinematics
        self.kin_model = CudaRobotModel(robot_cfg.kinematics)

        # update obstacles and stuff
    
    def plan(self, jp, jv, jn, goal, mode="joint_pos"):

        cu_js = JointState(
            position=self.tensor_args.to_device(jp),
            velocity=self.tensor_args.to_device(jv) * 0.0,
            acceleration=self.tensor_args.to_device(jv)* 0.0,
            jerk=self.tensor_args.to_device(jv) * 0.0,
            joint_names=jn
        )
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        goal_pos = torch.tensor(goal[:, 0:3], device=self.device)
        goal_orientation = torch.tensor(goal[:, 3:7], device=self.device)

        #     joint_pos_des
        #compute curobo sol
        ik_goal = Pose(
            position=goal_pos,
            quaternion=goal_orientation
        )
        self.plan_config.pose_cost_metric = None
        # result = self.motion_gen.plan_single(cu_js, ik_goal[0], self.plan_config)
        result = self.motion_gen.plan_batch_env(cu_js, ik_goal, self.plan_config.clone())
        traj = result.get_paths()
        if mode == "joint_pos":
            return traj
        elif mode == "ee_pose":
            ee_trajs = [self.kin_model.get_state(t.position) for t in traj]
            
            return ee_trajs
        else:
            raise ValueError("Invalid mode...")
    
    def pad_and_format(self, trajs):
            # pad out all
            trajs = [torch.cat((traj.ee_position, traj.ee_quaternion), dim=1) for traj in trajs]
            timesteps = max([len(traj) for traj in trajs])

            padded_tensors = []
            for traj in trajs:
                pad_val = traj[-1]
                padded_tensor = pad_val.repeat(timesteps, 1)
                padded_tensor[:traj.shape[0]] = traj
                padded_tensors.append(padded_tensor)

            padded_tensors = torch.stack(padded_tensors).to(self.device).permute(1,0,2)

            return padded_tensors
    
    def fk(self, jp):
        return self.kin_model.get_state(jp)