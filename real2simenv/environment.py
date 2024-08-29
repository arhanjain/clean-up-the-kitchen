import torch
import pickle
import gymnasium as gym
import numpy as np
from config.config import Config
import omni.isaac.lab.utils.math as math

from .utils import misc_utils
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.envs import ManagerBasedRLEnv
from typing import Optional
from omni.isaac.lab.utils.math import subtract_frame_transforms



class Real2SimRLEnv(ManagerBasedRLEnv):
    def __init__(self, custom_cfg: Config, *args, **kwargs):
        self.custom_cfg = custom_cfg
        super().__init__(*args, **kwargs)


    # TODO: this is a hacky solution which steps on reset to rerender camera,
    # There must be a better way to do this
    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)

        # get 0 action
        action = None
        if self.custom_cfg.actions.type == "absolute":
            pos = self.scene["ee_frame"].data.target_pos_source[:, 0]
            quat = self.scene["ee_frame"].data.target_quat_source[:, 0]
            gripper = torch.zeros((self.num_envs, 1), dtype=torch.float32).to(self.device)
            action = torch.cat((pos, quat, gripper), dim=1)
        else:
            raise NotImplementedError
        
        obs, rew, done, trunc, info = self.step(action)
        return obs, info


    def get_joint_info(self):
        '''
        Get joint information of the Franka robot entity in the scene,
        for all N environemnts.

        Returns
        -------
        joint_pos : torch.Tensor((N, 7)), dtype=torch.float32)
            Joint positions of the robot entity in the scene.
        joint_vel : torch.Tensor((N, 7)), dtype=torch.float32)
            Joint velocities of the robot entity in the scene.
        joint_names : list
            List of joint names of the robot entity in the scene.
        '''

        # Specify robot-specific parameters
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        robot_entity_cfg.resolve(self.scene)


        joint_pos = self.scene["robot"].data.joint_pos[:, robot_entity_cfg.joint_ids]
        joint_vel = self.scene["robot"].data.joint_vel[:, robot_entity_cfg.joint_ids]
        joint_names = self.scene["robot"].data.joint_names

        return joint_pos, joint_vel, joint_names

    def get_object_pose(self, object_name: str):
        pos_w = self.scene[object_name].data.root_pos_w
        quat_w = self.scene[object_name].data.root_quat_w

        # convert to robot frame
        robot_pos_w = self.scene["robot"].data.root_state_w[:, :3]
        robot_quat_w = self.scene["robot"].data.root_state_w[:, 3:7]

        pos_r, quat_r = subtract_frame_transforms(
            robot_pos_w, robot_quat_w,
            pos_w, quat_w
            )

        return pos_r, quat_r



    def get_camera_data(self):
        '''
        Get camera data for all N environments.

        Returns
        -------
        rgb : torch.Tensor((N, H, W, 3), dtype=torch.float32)
            RGB image data from the camera in the scene.
        seg : torch.Tensor((N, H, W), dtype=torch.uint8):
            Segmentation mask data from the camera in the scene.
        depth : torch.Tensor((N, H, W), dtype=torch.float32)
            Depth values per pixel from the camera in the scene.
        metadata : list
            List of metadata for each environment. Includes segmenation label map,
            camera intrinsics, camera pose, end-effector pose, and scene bounds.
        '''
        
        # RGB Image
        rgb = self.scene["camera"].data.output["rgb"]
        
        # Segmentation
        seg = self.scene["camera"].data.output["semantic_segmentation"].cpu().numpy()
        label_maps = []
        for info in self.scene["camera"].data.info:
            mapping = {}
            for k, v in info["semantic_segmentation"]["idToLabels"].items():
                mapping[v["class"]] = int(k)
            label_maps.append(mapping)
        
        # Depth values per pixel
        depth = self.scene["camera"].data.output["distance_to_image_plane"]

        # camera intrinsics
        intrinsics = self.scene["camera"].data.intrinsic_matrices

        # camera pose (extrinsics)
        cam_pos_w = self.scene["camera"].data.pos_w
        cam_quat_w = self.scene["camera"].data.quat_w_ros

        robot_pos_w = self.scene["robot"].data.root_state_w[:, :3]
        robot_quat_w = self.scene["robot"].data.root_state_w[:, 3:7]

        cam_pos_r, cam_quat_r = math.subtract_frame_transforms(
            robot_pos_w, robot_quat_w,
            cam_pos_w, cam_quat_w
        )
        cam_transformation = misc_utils.pos_and_quat_to_matrix(cam_pos_r, cam_quat_r)
        ee_frame = self.scene["ee_frame"]

        ee_transfomration = misc_utils.pos_and_quat_to_matrix(ee_frame.data.target_pos_source[:,0], ee_frame.data.target_quat_source[:, 0])
        
        # Assemble metadata
        metadata = []
        for i in range(self.num_envs):
            m = {
                    "intrinsics": intrinsics[i].cpu().numpy(),
                    "camera_pose": cam_transformation[i],
                    "ee_pose": ee_transfomration[i],
                    "label_map": label_maps[i],
                    "scene_bounds": [-0.4, -0.8, -0.2, 1.2, 0.8, 0.6],
                }
            metadata.append(m)

        # For debugging purposes
        # save_dir = "./data"
        # everything should be numpy
        # Remove 4th channel for rgb
        rgb = rgb[..., :-1].cpu().numpy()
        seg = seg
        depth = depth.cpu().numpy()

        # np.save(f"{save_dir}/rgb.npy", rgb)        
        # np.save(f"{save_dir}/mask.npy", seg)
        # np.save(f"{save_dir}/depth.npy", depth)
        # with open(f"{save_dir}/meta_data.pkl", "wb") as f:
        #     pickle.dump(metadata, f)

        return rgb, seg, depth, metadata

