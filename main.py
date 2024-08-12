import argparse
from omni.isaac.lab.app import AppLauncher
from wrappers.logger import wrap_env_in_logger

parser = argparse.ArgumentParser(description="test")
parser.add_argument(
    "--cpu", action="store_true", default=False, help="Use CPU pipeline."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

########################################

import torch
import gymnasium as gym
import real2simenv
from omni.isaac.lab_tasks.utils import parse_env_cfg
# from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg

from omegaconf import OmegaConf

from planning.orchestrator import Orchestrator
from omni.isaac.lab_tasks.utils import parse_env_cfg
import yaml

def do_nothing(env):
    ee_frame_sensor = env.unwrapped.scene["ee_frame"]
    tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
    tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
    gripper = torch.ones(env.unwrapped.num_envs, 1).to(env.unwrapped.device)
    action = torch.cat([tcp_rest_position, tcp_rest_orientation, gripper], dim=-1)
    for _ in range(10):
        env.step(action)

def main():
    # Load configuration
    with open("config.yml", "r") as file:
        cfg = OmegaConf.create(yaml.safe_load(file))
        # Attach USD info
        with open(cfg.usd_info_path, "r") as usd_info_file:
            cfg.usd_info = yaml.safe_load(usd_info_file)

    # create environment configuration
    env_cfg: real2simenv.Real2SimCfg = parse_env_cfg(
        args_cli.task,
        use_gpu=not args_cli.cpu,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.setup(cfg.usd_info)

    # video wrapper stuff
    viewer_cfg = cfg.video.copy()
    env_cfg.viewer.resolution = viewer_cfg.pop("viewer_resolution")
    env_cfg.viewer.eye = viewer_cfg.pop("viewer_eye")
    env_cfg.viewer.lookat = viewer_cfg.pop("viewer_lookat")
    video_kwargs = viewer_cfg

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, custom_cfg=cfg, render_mode="rgb_array")

    # apply wrappers
    if cfg.video.enabled:          
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = wrap_env_in_logger(env)

    # Reset environment
    env.reset()

    # Temp fix to image rendering, so that it captures RGB correctly before entering.
    do_nothing(env)
    env.reset()


    orchestrator = Orchestrator(env, cfg)
    plan_template = [
        ("grasp", {"target":"bowl"}),
    ]

    # Simulate environment
    while simulation_app.is_running():
        do_nothing(env)
        full_plan = orchestrator.generate_plan_from_template(plan_template)

        # ignoring using torch inference mode for now
        for segment in full_plan:
            env.step(segment)
        env.reset()
    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
