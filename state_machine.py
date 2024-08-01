import argparse
from math import comb



from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="hello hello hellooh")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments 
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
from omegaconf import OmegaConf
from collections.abc import Sequence
from m2t2.m2t2 import M2T2

import warp as wp
import numpy as np

import omni.isaac.lab_tasks  # noqa: F401
from numpy import dtype
#from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
from planner import MotionPlanner
from customenv import TestWrapper
from grasp import load_and_predict, visualize, m2t2_grasp_to_pos_and_quat
import grasp

# initialize warp
wp.init()

plan = [
    ("pregrasp", "movement"

]

class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    PLAN_GRASP = wp.constant(1)
    # APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)
    DONE = wp.constant(5)
    # APPROACH_GOAL = wp.constant(4)
    # UNGRASP_OBJECT = wp.constant(5)


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.3)
    APPROACH_GOAl = wp.constant(1.0)
    UNGRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(0.5)




@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    curr_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    des_gripper_state: wp.array(dtype=float),
    # plan: wp.array(dtype=float),
    plan_idx: wp.array(dtype=int),
    plan_length: wp.array(dtype=int),
    offset: wp.array(dtype=wp.transform),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]

    # decide next state
    if state == PickSmState.REST:
        des_ee_pose[tid] = curr_pose[tid]
        des_gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.PLAN_GRASP
            sm_wait_time[tid] = 0.0

    elif state == PickSmState.PLAN_GRASP:
        if plan_length[tid] == 0:
            des_ee_pose[tid] = curr_pose[tid] # wait till plan is generated!
        else:
            sm_state[tid] = PickSmState.APPROACH_OBJECT

    elif state == PickSmState.APPROACH_OBJECT:
        if plan_idx[tid] >= plan_length[tid]: # finished motion plan traj
            sm_state[tid] = PickSmState.GRASP_OBJECT
            sm_wait_time[tid] = 0.0
        else: # go to next motion plan step
            plan_idx[tid] += 1
            # des_ee_pose[tid] = plan[tid][plan_idx[tid]]
            des_gripper_state[tid] = GripperState.OPEN

    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = curr_pose[tid]
        des_gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0

    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], curr_pose[tid])
        des_gripper_state[tid] = GripperState.CLOSE
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.DONE
            sm_wait_time[tid] = 0.0

    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_OBJECT: The robot moves above the object.
    3. APPROACH_OBJECT: The robot moves to the object.
    4. GRASP_OBJECT: The robot grasps the object.
    5. LIFT_OBJECT: The robot lifts the object to the desired pose. This is the final state.
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", planner: MotionPlanner = None):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.planner = planner
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # plan
        self.plan = torch.zeros((self.num_envs, 500, 7), dtype=torch.float32, device=self.device)
        self.plan_length = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        self.plan_idx = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.5
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)
        self.plan_idx_wp = wp.from_torch(self.plan_idx, wp.int32)
        self.plan_length_wp = wp.from_torch(self.plan_length, wp.int32)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0
        self.plan_idx[env_ids] = 0
        self.plan_length[env_ids] = 0

    
    def do_plan(self, env, grasp_tools, env_ids):
        '''
        Generates grasping plan. Takes in gym environment,
        grasp tools (grasp_model, grasp_cfg), and sequence 
        of environment ids.
        '''
        # ENV IDS IS A TODO
        # predict grasp and motion plan
        if len(env_ids) == 0:
            return
        rgb, seg, depth, metadata = env.get_camera_data()
        rgb, seg, depth, metadata = rgb[env_ids], seg[env_ids], depth[env_ids], [metadata[i] for i in env_ids]
        loaded_data = rgb, seg, depth, metadata
        grasp_model, grasp_cfg = grasp_tools

        data, outputs = load_and_predict(loaded_data, grasp_model, grasp_cfg, obj_label="obj")
        visualize(grasp_cfg, data[0], {k: v[0] for k, v in outputs.items()})
        (goal_pos, goal_quat), success = grasp.choose_grasp(outputs)
        
        # breakpoint()
        #use success to filter what to send into planning
        if not np.any(success):
            return
        env_ids = env_ids[success]

        # only plan on successful grasps
        jp, jv, jn = env.get_joint_info()
        goal = torch.cat([goal_pos, goal_quat], dim=1)
        try:
            plan, success = self.planner.plan(jp, jv, jn, goal, mode="ee_pose")
        except:
            breakpoint()
        if not success: # TODO if we can split this per success?
            return
    
        # set the plans for the subset that is successful
        plan, plan_length = self.planner.test_format(plan, maxpad=500)
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        plan = plan[:, :, [0, 1, 2, 4, 5, 6, 3]]

        # TODO this will have different padding than the previous planning, we need a way to unify it
        self.plan[env_ids] = plan
        self.plan_length = plan_length
        self.plan_length_wp = wp.from_torch(self.plan_length.contiguous(), dtype=wp.int32)
        # self.plan_wp = wp.from_torch(self.plan.contiguous())
        # TODO ^ SO FAR THIS IS ONLY FOR PICKING THE OBJECT


    def compute(self, env, grasp_tools, curr_pose):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        curr_pose = curr_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        # convert to warp
        curr_pose_wp = wp.from_torch(curr_pose.clone().contiguous(), wp.transform)

        # plan for envs without a plan, if its time to plan
        self.do_plan(env, grasp_tools, torch.nonzero(self.plan_length == 0).cpu().numpy().squeeze())

        # use plan if not moved past end
        self.des_ee_pose = self.plan[range(self.num_envs), self.plan_idx]
        self.des_ee_pose = torch.where(
                (self.plan_idx < self.plan_length)[:, None],
                self.plan[range(self.num_envs), self.plan_idx],
                self.des_ee_pose
                )


        print("All envs states:", self.sm_state)
        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                curr_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                # self.plan_wp,
                self.plan_idx_wp,
                self.plan_length_wp,
                self.offset_wp,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        print("des pose", des_ee_pose)
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
    # parse configuration
    import customenv
    from configuration import GeneralCfg
    general_cfg = GeneralCfg().to_dict()
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Cube-Test-v0",
        use_gpu=not args_cli.cpu,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.setup(general_cfg)
    # create environment
    env = gym.make("Cube-Test-v0", cfg=env_cfg)
    env = TestWrapper(env)

    # motion planner
    planner = MotionPlanner(env)
    # grasp predictor
    grasp_cfg = OmegaConf.load("./grasp_config.yaml")
    grasp_model = M2T2.from_config(grasp_cfg.m2t2)
    ckpt = torch.load(grasp_cfg.eval.checkpoint)
    grasp_model.load_state_dict(ckpt["model"])
    grasp_model = grasp_model.cuda().eval()
    # create state machine
    pick_sm = PickAndLiftSm(
                env_cfg.sim.dt * env_cfg.decimation,
                env.unwrapped.num_envs, 
                env.unwrapped.device, 
                planner
            )

    # reset environment at start
    env.reset()
    for _ in range(10):
        env.step(torch.tensor(env.action_space.sample()).to(env.device))
    env.reset()
    pick_sm.reset_idx(range(env.unwrapped.num_envs))

    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = env.step(actions)[-2]

            # observations
            # -- end-effector frame
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()

        # advance state machine
        actions = pick_sm.compute(
            env, (grasp_model, grasp_cfg), 
            torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
        )

        # reset state machine
        if dones.any():
            # maybe reset the action if we get a done?
            pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
