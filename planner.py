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

from grasp import Grasper

    
class MotionPlanner:
    def __init__(self, env, grasper: Grasper):
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
            # usd_help.add_world_to_stage(world_cfg, base_frame=f"/World/world_{i}")
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

        # usd_help.add_world_to_stage(world_cfg, base_frame="/World")

        self.device = env.device

        # Forward kinematics
        self.kin_model = CudaRobotModel(robot_cfg.kinematics)

        self.env = env
        self.grasper = grasper

        # update obstacles and stuff
    def build_plan_from_template(self, plan_template):
        for action_type, object, location in plan_template:
            rgb, seg, depth, meta_data = self.env.get_camera_data()
            loaded_data = rgb, seg, depth, meta_data
            match action_type:
                case "move":
                    # get grasp pose
                    success = torch.zeros(self.env.num_envs)
                    while not torch.all(success):
                        grasp_pose, success = self.grasper.get_grasp(loaded_data, object)
                    pregrasp_pose = self.grasper.get_pregrasp(grasp_pose, 0.1)

                    # go to pregrasp
                    joint_pos, joint_vel, joint_names = self.env.get_joint_info()
                    traj, success = self.plan(joint_pos, joint_vel, joint_names, pregrasp_pose, mode="ee_pose")
                    if not success:
                        print("Failed to plan to pregrasp")
                        yield None
                    else:
                        traj, traj_length = self.test_format(traj, maxpad=max(t.ee_position.shape[0] for t in traj))
                        yield torch.cat((traj, torch.ones(self.env.num_envs, traj.shape[1], 1).to(self.device)), dim=2)

                    # go to grasp
                    joint_pos, joint_vel, joint_names = self.env.get_joint_info()
                    traj, success = self.plan(joint_pos, joint_vel, joint_names, grasp_pose, mode="ee_pose")
                    if not success:
                        print("Failed to plan to grasp")
                        yield None  
                    else:
                        traj, traj_length = self.test_format(traj, maxpad=max(t.ee_position.shape[0] for t in traj))
                        # traj, traj_length = self.test_format(traj, maxpad=500)
                        yield torch.cat((traj, torch.ones(self.env.num_envs, traj.shape[1], 1).to(self.device)), dim=2)

                    # grasp
                    ee_frame_sensor = self.env.unwrapped.scene["ee_frame"]
                    tcp_rest_position = ee_frame_sensor.data.target_pos_source[..., 0, :].clone()
                    tcp_rest_orientation = ee_frame_sensor.data.target_quat_source[..., 0, :].clone()
                    ee_pose = torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1)
                    close_gripper = -1 * torch.ones(self.env.num_envs, 1).to(self.device)
                    yield torch.cat((ee_pose, close_gripper), dim=1).repeat(1, 10, 1)

                    # go to pregrasp
                    joint_pos, joint_vel, joint_names = self.env.get_joint_info()
                    traj, success = self.plan(joint_pos, joint_vel, joint_names, pregrasp_pose, mode="ee_pose")
                    if not success:
                        print("Failed to plan to pregrasp")
                        yield None
                    else:
                        traj, traj_length = self.test_format(traj, maxpad=max(t.ee_position.shape[0] for t in traj))
                        yield torch.cat((traj, -1*torch.ones(self.env.num_envs, traj.shape[1], 1).to(self.device)), dim=2)


                case _:
                    raise ValueError("Invalid action type!")
    
    def plan(self, jp, jv, jn, goal, mode="joint_pos"):

        cu_js = JointState(
            position=self.tensor_args.to_device(jp),
            velocity=self.tensor_args.to_device(jv) * 0.0,
            acceleration=self.tensor_args.to_device(jv)* 0.0,
            jerk=self.tensor_args.to_device(jv) * 0.0,
            joint_names=jn
        )
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        goal_pos = goal[:, 0:3].clone().detach().to(self.device)
        goal_orientation = goal[:, 3:7].clone().detach().to(self.device)
        #     joint_pos_des
        #compute curobo sol
        ik_goal = Pose(
            position=goal_pos,
            quaternion=goal_orientation
        )
        self.plan_config.pose_cost_metric = None

        # Differentiate between single and batch planning
        result = None
        if len(ik_goal) == 1:
            result = self.motion_gen.plan_single(cu_js, ik_goal[0], self.plan_config)
            # if mode == "joint_pos":
            #     return traj, True
            # elif mode == "ee_pose":
            #     return [self.kin_model.get_state(traj[0].position)], True
        else: 
            result = self.motion_gen.plan_batch_env(cu_js, ik_goal, self.plan_config.clone())

        # Check failure cases
        if result.status == MotionGenStatus.TRAJOPT_FAIL:
            print('TRAJOPT_FAIL')
            return None, False
        if result.status == MotionGenStatus.IK_FAIL:
            print("IK FAILURE")
            return None, False

        # Extract the trajectories based on single vs batch
        traj = [result.interpolated_plan] if len(ik_goal) == 1 else result.get_paths()

        # Return in desired format
        if mode == "joint_pos":
            return traj, True
        elif mode == "ee_pose":
            ee_trajs = [self.kin_model.get_state(t.position) for t in traj]
            return ee_trajs, True
        else:
            raise ValueError("Invalid mode...")

    def test_format(self, trajs, maxpad=500):
            # pad out all
            trajs = [torch.cat((traj.ee_position, traj.ee_quaternion), dim=1) for traj in trajs]
            lengths = [len(traj) for traj in trajs]
            max_time = maxpad
            padded_tensors = []
            for traj in trajs:
                pad = torch.zeros(max_time - len(traj), 7).to(self.device)
                padded = torch.cat((traj, pad), dim=0)
                padded_tensors.append(padded)
            ends = torch.tensor(lengths, dtype=torch.int32).to(self.device) - 1
            return torch.stack(padded_tensors), ends
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
