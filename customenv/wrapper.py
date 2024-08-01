import torch
import pickle
import numpy as np
import omni.isaac.lab.utils.math as math

from . import mdp
from .utils import misc_utils
from gymnasium import Wrapper
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.envs import ManagerBasedRLEnv
from PIL import Image

class TestWrapper(Wrapper):
    def __init__(self, env: ManagerBasedRLEnv):
        if not isinstance(env.unwrapped, ManagerBasedRLEnv):
            raise ValueError("Environment must be a ManagerBasedRLEnv...")

        # self.env = env 
        # self.scene = env.scene
        super().__init__(env)
    
    def get_joint_info(self):
        # Specify robot-specific parameters
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        robot_entity_cfg.resolve(self.scene)


        joint_pos = self.scene["robot"].data.joint_pos[:, robot_entity_cfg.joint_ids]
        joint_vel = self.scene["robot"].data.joint_vel[:, robot_entity_cfg.joint_ids]
        joint_names = self.scene["robot"].data.joint_names

        return joint_pos, joint_vel, joint_names

    def get_camera_data(self):
        
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

        # Assemble metadata
        intrinsics = self.scene["camera"].data.intrinsic_matrices
        # camera pose
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
        save_dir = "./data"
        # everything should be numpy
        # Remove 4th channel for rgb
        rgb = rgb[..., :-1].cpu().numpy()
        seg = seg
        depth = depth.cpu().numpy()

        np.save(f"{save_dir}/rgb.npy", rgb)        
        np.save(f"{save_dir}/mask.npy", seg)
        np.save(f"{save_dir}/depth.npy", depth)
        with open(f"{save_dir}/meta_data.pkl", "wb") as f:
            pickle.dump(metadata, f)

        return rgb, seg, depth, metadata


    def goal_pose(self):
        return mdp.generated_commands(self.env, "object_pose")
