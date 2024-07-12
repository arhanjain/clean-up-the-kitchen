import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
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
    def __init__(self, scene, device):
        usd_help = UsdHelper()
        self.tensor_args = TensorDeviceType()

        robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
        world_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        )

        trajopt_dt = None
        optimize_dt = True
        trajopt_tsteps = 16
        trim_steps = None
        max_attempts = 4
        interpolation_dt = 0.05
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            self.tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            interpolation_dt=interpolation_dt,
            collision_cache={"obb": 30, "mesh": 100},
            optimize_dt=optimize_dt,
            trajopt_dt=trajopt_dt,
            trajopt_tsteps=trajopt_tsteps,
            trim_steps=trim_steps,
        )
        self.motion_gen = MotionGen(motion_gen_config)
        print("warming up...")
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False, parallel_finetune=True)
        print("Curobo is ready!")

        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=2,
            max_attempts=max_attempts,
            enable_finetune_trajopt=True,
            parallel_finetune=True,
        )

        usd_help.load_stage(scene.stage)
        usd_help.add_world_to_stage(world_cfg, base_frame="/World")

        self.device = device

        # update obstacles and stuff
    
    def plan(self, jp, jv, jn, goal):

        cu_js = JointState(
            position=self.tensor_args.to_device(jp),
            velocity=self.tensor_args.to_device(jv),
            acceleration=self.tensor_args.to_device(jv)* 0.0,
            jerk=self.tensor_args.to_device(jv) * 0.0,
            joint_names=jn
        )
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        goal_pos = torch.tensor(goal[0:3], device=self.device)
        goal_orientation = torch.tensor(goal[3:7], device=self.device)

        #     joint_pos_des
        #compute curobo sol
        ik_goal = Pose(
            position=goal_pos,
            quaternion=goal_orientation
        )
        self.plan_config.pose_cost_metric = None
        # result = self.motion_gen.plan_single(cu_js, ik_goal.unsqueeze(0), self.plan_config)
        result = self.motion_gen.plan_single(cu_js, ik_goal[0], self.plan_config)
        # breakpoint()
        traj = result.get_interpolated_plan()

        return traj
        # info = self.motion_gen.kinematics.forward(traj.position)
        # ee_pos, ee_quat = info[0], info[1]


# def move_to_cartesian_pose(
#     target_pose,
#     gripper,
#     motion_planner,
#     controller,
#     env,
#     progress_threshold=1e-3,
#     max_iter_per_waypoint=20,
# ):

#     controller.reset()

#     start = env.unwrapped._robot.get_ee_pose().copy()
#     start = np.concatenate((start[:3], euler_to_quat_mujoco(start[3:])))
#     target_pose = target_pose.copy()

#     if target_pose[5] > np.pi / 2:
#         target_pose[5] -= np.pi
#     if target_pose[5] < -np.pi / 2:
#         target_pose[5] += np.pi

#     goal = np.concatenate((target_pose[:3], euler_to_quat_mujoco(target_pose[3:])))
#     qpos_plan = motion_planner.plan_motion(start, goal, return_ee_pose=True)

#     steps = 0
#     imgs = []

#     # first waypoint is current pose -> start from i=1
#     for i in range(len(qpos_plan.ee_position)-1):
#         des_pose = np.concatenate(
#             (
#                 qpos_plan.ee_position[i+1].cpu().numpy(),
#                 quat_to_euler_mujoco(qpos_plan.ee_quaternion[i].cpu().numpy()),
#             )
#         )
#         last_curr_pose = env.unwrapped._robot.get_ee_pose()

#         for j in range(max_iter_per_waypoint):

#             # get current pose
#             curr_pose = env.unwrapped._robot.get_ee_pose()

#             # run PD controller
#             act = controller.update(curr_pose, des_pose)
#             act = np.concatenate((act, [gripper]))

#             # step env
#             obs, _, _, _ = env.step(act)
#             steps += 1
         

#             image = obs[f"front_rgb"]
    
#             # import cv2
#             # cv2.imshow('Real-time video', cv2.cvtColor(image,
#             #                                            cv2.COLOR_BGR2RGB))

#             # # Press 'q' on the keyboard to exit the loop
#             # if cv2.waitKey(1) & 0xFF == ord('q'):
#             #     break

#             curr_pose = env.unwrapped._robot.get_ee_pose()
#             pos_diff = curr_pose[:3] - last_curr_pose[:3]
#             angle_diff = curr_pose[3:] - last_curr_pose[3:]
#             angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
#             err = np.linalg.norm(pos_diff) + np.linalg.norm(angle_diff)

#             # early stopping when actions don't change position anymore
#             # 5x more accuracy for last 3 steps
#             # if i > len(qpos_plan.ee_position) - 3:
#             #     if err < progress_threshold / 5:
#             #         break
#             # elif err < progress_threshold:
#             #     break
#             if err < progress_threshold:
#                 break

#             last_curr_pose = curr_pose

#     return imgs, steps