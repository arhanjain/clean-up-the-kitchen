# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Tuple

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.sensors import FrameTransformer
# from real2simenv.utils import misc_utils
from cleanup.real2simenv.utils import misc_utils

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return torch.cat((object_pos_b, object_quat_b), dim=1)

def get_camera_data(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    type: str = "rgb"
) -> torch.Tensor:
  
    camera = env.scene[camera_cfg.name]
    return camera.data.output[type][0][..., :3]

def get_point_cloud(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
) -> torch.Tensor:
    intrinsics = env.scene[camera_cfg.name].data.intrinsic_matrices
    depth = env.scene[camera_cfg.name].data.output["distance_to_image_plane"]

    raw_pcd = misc_utils.depth_to_xyz(depth, intrinsics)
    return raw_pcd[0]
    # mask = torch.isfinite(raw_pcd).all(dim=-1)
    # filtered_pcd = raw_pcd[mask]
    # return filtered_pcd.view(-1, 3)



def gripper_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    gripper_state = robot.data.joint_pos[:, -2:]
    # TODO: convert [a,b] to [-1, 1]
    # torch.max(gripper_state, torch.tensor(0.0))
    return gripper_state

def ee_pose(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w

    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot_pos_w, robot_quat_w,
        ee_pos_w, ee_quat_w
    )

    return torch.cat((ee_pos_b, ee_quat_b), dim=1)
