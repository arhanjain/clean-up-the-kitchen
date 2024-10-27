import argparse
parser = argparse.ArgumentParser(description="test")
# parser.add_argument(
#     "--disable_fabric", action="store_true",
#     default=False, help="Disable fabric and use USD I/O operations.",) 
# parser.add_argument(
#     "--num_envs", type=int, default=1, help="Number of environments to simulate."
# )
# parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--ds_name", type=str, required=True, help="Name of the dataset.")

args = parser.parse_args()

########################################

# import torch
import os
import time
import pickle
import hydra 
import numpy as np
import cv2
import gymnasium as gym
from pathlib import Path

# from cleanup.config import Config
from droid.robot_env import RobotEnv
from droid.misc.time import time_ms
from droid.controllers.oculus_controller import VRPolicy
from droid.trajectory_utils.trajectory_writer import TrajectoryWriter

def collect_trajectory(
    env,
    controller=None,
    policy=None,
    horizon=None,
    save_filepath=None,
    metadata=None,
    wait_for_controller=False,
    obs_pointer=None,
    save_images=False,
    recording_folderpath=False,
    randomize_reset=False,
    reset_robot=True,
):
    """
    Collects a robot trajectory.
    - If policy is None, actions will come from the controller
    - If a horizon is given, we will step the environment accordingly
    - Otherwise, we will end the trajectory when the controller tells us to
    - If you need a pointer to the current observation, pass a dictionary in for obs_pointer
    """

    # Check Parameters #
    assert (controller is not None) or (policy is not None)
    assert (controller is not None) or (horizon is not None)
    if wait_for_controller:
        assert controller is not None
    if obs_pointer is not None:
        assert isinstance(obs_pointer, dict)
    if save_images:
        assert save_filepath is not None

    # Reset States #
    if controller is not None:
        controller.reset_state()
    # env.camera_reader.set_trajectory_mode()

    # Prepare Data Writers If Necesary #
    if save_filepath:
        # traj_writer = TrajectoryWriter(save_filepath, metadata=metadata, save_images=save_images, exists_ok=True)
        observations = []
        actions = []

    if recording_folderpath:
        env.camera_reader.start_recording(recording_folderpath)

    # Prepare For Trajectory #
    num_steps = 0
    if reset_robot:
        env.reset(randomize=randomize_reset)

    # Begin! #
    print("Begin :)")
    while True:
        # Collect Miscellaneous Info #
        controller_info = {} if (controller is None) else controller.get_info()
        skip_action = wait_for_controller and (not controller_info["movement_enabled"])
        control_timestamps = {"step_start": time_ms()}

        # Get Observation #
        obs = env.get_observation()

        # display img
        imgs =obs["image"]
        for k, img in imgs.items():
            cv2.imshow(k, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        

        if obs_pointer is not None:
            obs_pointer.update(obs)
        obs["controller_info"] = controller_info
        obs["timestamp"]["skip_action"] = skip_action

        # Get Action #
        control_timestamps["policy_start"] = time_ms() 
        if policy is None:
            action, controller_action_info = controller.forward(obs, include_info=True)
        else:
            action = policy.forward(obs)
            controller_action_info = {}

        # Regularize Control Frequency #
        control_timestamps["sleep_start"] = time_ms()
        comp_time = time_ms() - control_timestamps["step_start"]
        sleep_left = (1 / env.control_hz) - (comp_time / 1000)
        if sleep_left > 0:
            time.sleep(sleep_left)

        # Moniter Control Frequency #
        # moniter_control_frequency = True
        # if moniter_control_frequency:
        # 	print('Sleep Left: ', sleep_left)
        # 	print('Feasible Hz: ', (1000 / comp_time))

        # Step Environment #
        skip_timestep_save = False
        control_timestamps["control_start"] = time_ms()
        if skip_action:
            action_info = env.create_action_dict(np.zeros_like(action))
            skip_timestep_save = True
        else:
            action_threshold = 1e-3
            if np.all(np.abs(action[:-1]) < action_threshold):
                action_info = env.create_action_dict(np.zeros_like(action))
                skip_timestep_save = True
            else:
                print(action)
                action_info = env.step(action)
        action_info.update(controller_action_info)

        # Save Data #
        control_timestamps["step_end"] = time_ms()
        obs["timestamp"]["control"] = control_timestamps
        timestep = {"observation": obs, "action": action_info}
        if save_filepath and not skip_timestep_save:
            # traj_writer.write_timestep(timestep)

            # make obs
            ee_pose = obs["robot_state"]["cartesian_position"]
            gripper_state = obs["robot_state"]["gripper_position"]
            imgs = list(obs["image"].values())
            rgb = np.concatenate(imgs, axis=1)

            latest_obs = {
                "policy": {
                    "ee_pose": ee_pose,
                    "gripper_state": np.array([gripper_state]),
                    "rgb": rgb
                }
            }
            observations.append(latest_obs)
            actions.append(action)


        # Check Termination #
        num_steps += 1
        if horizon is not None:
            end_traj = horizon == num_steps
        else:
            end_traj = controller_info["success"] or controller_info["failure"]

        # Close Files And Return #
        if end_traj:
            if recording_folderpath:
                env.camera_reader.stop_recording()
            if save_filepath:
                # traj_writer.close(metadata=controller_info)
                data_dict = {
                        "observations": observations,
                        "actions": actions,
                        "rewards": [],
                        "dones": [],
                        "truncateds": [],
                        "next_observations": [],
                        }
                np.savez(save_filepath, **data_dict)

            if controller_info["failure"]:
                os.remove(save_filepath)
            return controller_info

def get_last_trajectory(dir):
    trajs = list(dir.glob("*.npz"))
    if len(trajs) == 0:
        return 0
    return max([int(str(traj).split("_")[-1].split(".")[0]) for traj in trajs]) + 1

# @hydra.main(version_base=None, config_path="./cleanup/config", config_name="config")
def main():#(cfg: Config):

    env = RobotEnv(action_space="cartesian_velocity", gripper_action_space="position")
    teleop = VRPolicy(
            pos_action_gain=8,
            rot_action_gain=3,
            )

    ds_dir = Path(f"./data/{args.ds_name}/")
    ds_dir.mkdir(exist_ok=True)
    
    info = {
        "action_space": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
        "obs_space": gym.spaces.Dict({
            "policy": gym.spaces.Dict({
                "ee_pose": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "gripper_state": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "rgb": gym.spaces.Box(0, 255, shape=(244, 244, 3), dtype=np.uint8),
            })
        })
    }
    with open(str(ds_dir/"info.pkl"), "wb") as f:
        pickle.dump(info, f)


    episode = get_last_trajectory(ds_dir)
    running = True
    while True:
        print(f"Press 'A' to Start Collecting, 'B' to Stop")
        while True:
            info = teleop.get_info()
            if info["success"]:
                break 
            elif info["failure"]:
                env.close()
                running = False
                break
        if not running:
            break

        print(f"Starting collection of episode {episode}!")
        controller_info = collect_trajectory(
                env, 
                controller=teleop, 
                save_filepath=str(ds_dir / f"episode_{episode}.npz"),
                save_images=True,
                wait_for_controller=True,
                )
        if controller_info["success"]:
            episode += 1
        env.reset(randomize=True)
    env.close()




if __name__ == "__main__":
    # run the main function
    main()
