import torch
import numpy as np
import sys
sys.path.append('/home/jacob/projects/clean-up-the-kitchen/anygrasp_sdk/grasp_detection')
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

class Grasper:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.anygrasp = AnyGrasp(cfg.anygrasp)
        self.anygrasp.load_net()

        # Tune max_gripper_width to be within a valid range
        self.cfg.anygrasp.max_gripper_width = max(0, min(0.1, self.cfg.anygrasp.max_gripper_width))
    
    def get_grasp(self, env):
        rgb_batch, seg_batch, depth_batch, meta_data_batch = env.get_camera_data()
        batch_size = rgb_batch.shape[0]
        grasp_poses = []
        successes = []

        for i in range(batch_size):
            rgb = rgb_batch[i] / 255.0  # Normalize
            depth = depth_batch[i]
            intrinsics = meta_data_batch[i]['intrinsics']
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]  # Focal lengths
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]  # Principal points
            scale = 1000.0  # Scale factor for depth

            # Set workspace limits (can be tuned)
            xmin, xmax = -0.19, 0.12
            ymin, ymax = 0.02, 0.15
            zmin, zmax = 0.0, 1.0
            lims = [xmin, xmax, ymin, ymax, zmin, zmax]

            # Get point cloud
            xmap, ymap = np.arange(depth.shape[1]), np.arange(depth.shape[0])
            xmap, ymap = np.meshgrid(xmap, ymap)
            points_z = depth / scale
            points_x = (xmap - cx) / fx * points_z
            points_y = (ymap - cy) / fy * points_z

            # Mask and crop the point cloud
            mask = (points_z > 0) & (points_z < 1)
            points = np.stack([points_x, points_y, points_z], axis=-1)
            points = points[mask].astype(np.float32)
            colors = rgb[mask].astype(np.float32)
            print(points.min(axis=0), points.max(axis=0))
            breakpoint()
            # Get grasps
            gg, cloud = self.anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

            if gg:
                gg = gg.nms().sort_by_score()
                top_grasp = gg[0]
                
                T = np.eye(4)
                T[:3, :3] = top_grasp.rotation
                T[:3, 3] = top_grasp.translation
                grasp_poses.append(torch.tensor(T, dtype=torch.float32))
                successes.append(True)
            else:
                grasp_poses.append(torch.zeros(4, 4, dtype=torch.float32))
                successes.append(False)
        breakpoint()
        grasp_poses = torch.stack(grasp_poses) if len(grasp_poses) > 0 else torch.zeros(batch_size, 4, 4)
        # grasp_poses = grasp_to_pos_and_quat()
        
        success_tensor = torch.tensor(successes, dtype=torch.bool)
        return grasp_poses, success_tensor

    def get_prepose(self, grasp_pose, offset):
        pos = grasp_pose[:, :3, 3]
        rot_matrix = grasp_pose[:, :3, :3]

        direction = torch.stack([
            2 * (rot_matrix[:, 0, 2] + rot_matrix[:, 1, 1]),
            2 * (rot_matrix[:, 1, 2] - rot_matrix[:, 0, 0]),
            1 - 2 * (rot_matrix[:, 0, 0] ** 2 + rot_matrix[:, 1, 1] ** 2)
        ], dim=-1)

        pregrasp_pos = pos - offset * direction
        pregrasp_pose = grasp_pose.clone()
        pregrasp_pose[:, :3, 3] = pregrasp_pos

        return pregrasp_pose

    @staticmethod
    def grasp_to_pos_and_quat(transform_mat):
        '''
        Converts a transformation matrix to position and quaternion format.

        Parameters
        ----------
        transform_mat: torch.tensor(float) shape: (N, 4, 4)
            The transformation matrix.

        Returns
        -------
        pos: torch.tensor(float) shape: (N, 3)
            The positions extracted from the transformation matrix.
        quat: torch.tensor(float) shape: (N, 4)
            The quaternions extracted from the transformation matrix.
        '''
        pos = transform_mat[..., :3, 3].clone()

        # Extract rotation matrix from the transformation matrix and convert to quaternion
        rotation_matrix = transform_mat[..., :3, :3]

        quat = math.quat_from_matrix(rotation_matrix)
        euler_angles = math.euler_xyz_from_quat(quat) 

        # Unpack the tuple
        roll, pitch, yaw = euler_angles

        yaw -= np.pi/2  # rotate to account for rotated frame between M3T2 and Isaac

        # Adjust the yaw angle
        yaw = torch.where(yaw > np.pi/2, yaw - np.pi, yaw)
        yaw = torch.where(yaw < -np.pi/2, yaw + np.pi, yaw)

        # Convert the adjusted Euler angles back to quaternion
        adjusted_quat = math.quat_from_euler_xyz(roll, pitch, yaw).view(-1, 4)

        return pos, adjusted_quat
