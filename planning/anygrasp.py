import torch
import numpy as np
import sys
sys.path.append('/home/jacob/projects/clean-up-the-kitchen/anygrasp_sdk/grasp_detection')
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
import omni.isaac.lab.utils.math as math

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
            scale = 10.0  # Scale factor for depth

            # Set workspace limits (can be tuned) or ignored
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
            # Get grasps
            gg, cloud = self.anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

            if gg:
                gg = gg.nms().sort_by_score()
                top_grasp = gg[0]
                
                T = np.eye(4)
                T[:3, :3] = top_grasp.rotation_matrix
                T[:3, 3] = top_grasp.translation
                grasp_poses.append(torch.tensor(T, dtype=torch.float32))
                successes.append(True)
            else:
                grasp_poses.append(torch.zeros(4, 4, dtype=torch.float32))
                successes.append(False)
        grasp_poses = torch.stack(grasp_poses) if len(grasp_poses) > 0 else torch.zeros(batch_size, 4, 4)
        grasp_poses = self.grasp_to_pos_and_quat(grasp_poses)
        
        success_tensor = torch.tensor(successes, dtype=torch.bool)
        breakpoint()
        return grasp_poses, success_tensor

    @staticmethod
    def get_prepose(pose, offset):
        '''
        Returns the pregrasp pose for a given grasp pose and offset
        Parameters
        ----------
        grasp_pose: torch.tensor(float) shape: (N, 7)
            The position and quaternion of the grasp pose
        offset: float
            The distance to move away from the original pose
        Returns
        -------
        pregrasp_pose: torch.tensor(float) shape: (N, 7)
            The position and quaternion of the pregrasp poses
        '''

        pos, quat = pose[:, :3], pose[:, 3:]

        # Normalize quaternions
        # norms = torch.norm(quat, dim=1, keepdim=True)
        # q = quat / norms
        q = quat / torch.norm(quat, dim=-1, keepdim=True)

        # Calculate direction vectors
        # direction = torch.empty((quat.shape[0], 3), device=quat.device)
        # direction[:, 0] = 2 * (q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3])
        # direction[:, 1] = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
        # direction[:, 2] = 1 - 2 * (q[:, 0]**2 + q[:, 1]**2)

        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        direction = torch.stack([
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x**2 + y**2)
        ], dim=-1)

        pregrasp = pose.clone()
        pregrasp[:, :3] -= offset * direction

        return pregrasp

    @staticmethod
    def grasp_to_pos_and_quat(transform_mat):
        '''
        Converts a transformation matrix to a 7-DOF pose (position and quaternion format).

        Parameters
        ----------
        transform_mat: torch.tensor(float) shape: (N, 4, 4)
            The transformation matrix.

        Returns
        -------
        pose: torch.tensor(float) shape: (N, 7)
            The 7-DOF pose consisting of positions and quaternions extracted from the transformation matrix.
        '''
        pos = transform_mat[..., :3, 3].clone()  # Extract position (3-DOF)

        # Extract rotation matrix from the transformation matrix and convert to quaternion (4-DOF)
        rotation_matrix = transform_mat[..., :3, :3]
        quat = math.quat_from_matrix(rotation_matrix)  # Convert rotation matrix to quaternion

        # Adjust the quaternion if needed (based on your specific requirements)
        euler_angles = math.euler_xyz_from_quat(quat)
        roll, pitch, yaw = euler_angles

        yaw -= np.pi / 2  # Adjust yaw to account for rotated frame

        # Ensure yaw is within the correct range
        yaw = torch.where(yaw > np.pi/2, yaw - np.pi, yaw)
        yaw = torch.where(yaw < -np.pi/2, yaw + np.pi, yaw)

        # Convert the adjusted Euler angles back to quaternion
        adjusted_quat = math.quat_from_euler_xyz(roll, pitch, yaw).view(-1, 4)

        # Concatenate position and quaternion to form the 7-DOF pose
        return torch.cat([pos, adjusted_quat], dim=-1)  # Shape: (N, 7)
