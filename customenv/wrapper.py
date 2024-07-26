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

        
        # Mask 
        seg = self.scene["camera"].data.output["semantic_segmentation"]
        mask = torch.clamp(seg-1, max=1).cpu().numpy().astype(np.uint8) * 255
        
        # Depth values per pixel
        depth = self.scene["camera"].data.output["distance_to_image_plane"]

        # Assemble metadata
        metadata = {}
        intrinsics = self.scene["camera"].data.intrinsic_matrices[0]

        # camera pose
        cam_pos_w = self.scene["camera"].data.pos_w
        cam_quat_w = self.scene["camera"].data.quat_w_ros

        robot_pos_w = self.scene["robot"].data.root_state_w[:, :3]
        robot_quat_w = self.scene["robot"].data.root_state_w[:, 3:7]

        cam_pos_r, cam_quat_r = math.subtract_frame_transforms(
            robot_pos_w, robot_quat_w,
            cam_pos_w, cam_quat_w
        )
        transformation = misc_utils.pos_and_quat_to_matrix(cam_pos_r, cam_quat_r)

        # filler from existing file
        ee_pose = np.array([[ 0.02123945,  0.82657526,  0.56242531,  0.18838109],
        [ 0.99974109, -0.02215279, -0.00519713, -0.01743025],
        [ 0.00816347,  0.56239007, -0.82683176,  0.6148137 ],
        [ 0.        ,  0.        ,  0.        ,  1.        ]])
        scene_bounds = np.array([-0.4, -0.8, -0.2, 1.2, 0.8, 0.6])

        metadata["intrinsics"] = intrinsics.cpu().numpy()
        metadata["camera_pose"] = transformation[0]
        metadata["ee_pose"] = ee_pose
        metadata["label_map"] = None

        # For debugging purposes
        save_dir = "/home/arhan/projects/IsaacLab/source/standalone/clean-up-the-kitchen/data"
        Image.fromarray(rgb[0].cpu().numpy()).convert("RGB").save(f"{save_dir}/rgb.png")
        Image.fromarray(mask[0], mode="L").save(f"{save_dir}/seg.png")
        np.save(f"{save_dir}/depth.npy", depth.cpu().numpy()[0])
        with open(f"{save_dir}/meta_data.pkl", "wb") as f:
            pickle.dump(metadata, f)

        return rgb[0][None], mask[0][None], depth[0][None], metadata


    def goal_pose(self):
        return mdp.generated_commands(self.env, "object_pose")