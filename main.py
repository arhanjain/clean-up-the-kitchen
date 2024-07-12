
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="test")
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

########################################

import torch
import numpy as np
import gymnasium as gym
import omni.isaac.lab.sim as sim_utils

import omni.isaac.lab_tasks
from omni.isaac.lab_tasks.utils import parse_env_cfg

from omni.isaac.lab.assets.asset_base_cfg import AssetBaseCfg
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG
from omni.isaac.lab.utils.math import subtract_frame_transforms

from planner import MotionPlanner

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 1.05))
    )

    #cube 1
    cube_one = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube1",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.3, 1.2))
    )
    # cube_two = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Cube2",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.2, 0.2, 0.2),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg()
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, -0.3, 0.25))
    # )

    # articulation
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.0, 0.0, 1.05)

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]

    #controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    
    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm
    ee_goals = [
        [0.5, 0, 0.7, 0.707, 0, 0.707, 0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    # Specify robot-specific parameters
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]


    sim_dt = sim.get_physics_dt()
    # Define simulation stepping
    count = 0
    # Simulation loop
    goals = [
        # [0.5, -0.5, 0.5, 0.707, 0, 0.707, 0],
        [0.5, -0.5, 0.7, 0.707, 0, 0.707, 0],
        [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
    ]
    planner = MotionPlanner(scene, sim.device)
    goal_idx = None
    goal = None
    plan = None
    #reset
    count = 0
    # reset joint state
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    # reset actions
    ik_commands[:] = ee_goals[current_goal_idx]
    joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
    # reset cont     
    diff_ik_controller.reset()
    diff_ik_controller.set_command(ik_commands)

    while simulation_app.is_running():
        # reset
        # if count % 150 == 0:
            # reset time
        # else:
            # # obtain quantities from simulation
            # jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            # ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            # root_pose_w = robot.data.root_state_w[:, 0:7]
            # joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # # compute frame in root frame
            # ee_pos_b, ee_quat_b = subtract_frame_transforms(
            #     root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            # )
            # # compute the joint commands
            # joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        # if count % 500 == 0:
            # switch goal
        # goal_idx = 0 if goal_idx and goal_idx != 0 else 1
        # goal = np.array(goals[goal_idx])
        block = scene["cube_one"].data.root_pos_w - robot.data.root_pos_w
        quat = torch.tensor([[0, 0.707,0.707,0]], device=sim.device)
        goal = torch.cat([block, quat], dim=1)

        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        joint_vel = robot.data.joint_vel[:, robot_entity_cfg.joint_ids]
        joint_names = robot.data.joint_names
        plan = planner.plan(joint_pos, joint_vel, joint_names, goal.squeeze())
        # print("NEW PLAN!")
        # print(plan.position)
        for joint_pos in plan.position:
            for j in range(20):
                robot.set_joint_position_target(joint_pos, joint_ids=robot_entity_cfg.joint_ids)
                scene.write_data_to_sim()
                # perform step
                sim.step()
                # update sim-time
                count += 1
                # update buffers
                scene.update(sim_dt)

                ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
                ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
                goal_marker.visualize(goal[:, 0:3], goal[:, 3:7])


        while(True):
            robot.set_joint_position_target(joint_pos, joint_ids=robot_entity_cfg.joint_ids)
            scene.write_data_to_sim()
            # perform step
            sim.step()
            # update sim-time
            count += 1
            # update buffers
            scene.update(sim_dt)

            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            goal_marker.visualize(goal[:, 0:3], goal[:, 3:7])




        # # else:
            
        # for joint_pos in traj.position:
            
        #     joint_pos_des = joint_pos
                
        #     for i in range(5):
        #         # apply actions
        #         robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        #         scene.write_data_to_sim()q
        #         # perform step
        #         sim.step()
        #         # update sim-time
        #         count += 1
        #         # update buffers
        #         scene.update(sim_dt)

        #         # obtain quantities from simulation
        #         ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        #         # update marker positions
        #         ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        #         goal_marker.visualize(ik_commands[:, 0:3] , ik_commands[:, 3:7])
        



def main():
    # sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    # sim = sim_utils.SimulationContext(sim_cfg)

    # scene_cfg = TableTopSceneCfg(num_envs=1, env_spacing=2.0)
    # scene = InteractiveScene(scene_cfg)
    # # Play the simulator
    # sim.reset()
    # # Now we are ready!
    # print("[INFO]: Setup complete...")
    # # Run the simulator
    # run_simulator(sim, scene)

     # create environment configuration
    import cleandakitchen
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
