import torch
import numpy as np
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

class Grasper:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.anygrasp = AnyGrasp(cfg.anygrasp)
        self.anygrasp.load_net()

        # (Tune, idk this info)
        self.cfg.anygrasp.max_gripper_width = max(0, min(0.1, self.cfg.anygrasp.max_gripper_width))
    
    def get_grasp(self, env, target):
        rgb_batch, seg_batch, depth_batch, meta_data_batch = env.get_camera_data()
        batch_size = rgb_batch.shape[0]
        grasp_poses = []
        successes = []

        for i in range(batch_size):
            rgb = rgb_batch[i]
            depth = depth_batch[i]
            intrinsics = meta_data_batch[i]['intrinsics']
            fx, fy = intrinsics['fx'], intrinsics['fy']
            cx, cy = intrinsics['cx'], intrinsics['cy']
            scale = 1000.0  # Tune

            # Set workspace and filter output grasps (probably need to tune)
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
            colors = rgb[mask].astype(np.float32) / 255.0

            # Get grasps
            gg, cloud = self.anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

            if len(gg) == 0:
                grasp_poses.append(None)
                successes.append(False)
            else:
                gg = gg.nms().sort_by_score()
                grasp_pose = gg[0].pose
                grasp_poses.append(torch.tensor(grasp_pose, dtype=torch.float32))
                successes.append(True)

        grasp_poses = torch.stack(grasp_poses) if grasp_poses[0] is not None else torch.zeros(batch_size, 4, 4)
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
