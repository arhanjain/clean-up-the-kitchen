import math
from PIL import Image
from sklearn.neighbors import KDTree
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import torch

from m2t2.dataset_utils import sample_points, depth_to_xyz, normalize_rgb
from m2t2.dataset import collate
from m2t2.m2t2 import M2T2
from m2t2.plot_utils import get_set_colors
from m2t2.train_utils import to_cpu, to_gpu
from m2t2.dataset_utils import denormalize_rgb, sample_points, jitter_gaussian
from m2t2.meshcat_utils import (
    create_visualizer, make_frame, visualize_grasp, visualize_pointcloud
)
import omni.isaac.lab.utils.math as math
from torchvision import transforms
def load_and_predict(cfg, meta_data, rgb, depth, seg):
    # We only want the outputs, not meta_data
    data = load_rgb_xyz(
        meta_data, rgb, depth, seg, cfg.data.robot_prob,
        cfg.data.world_coord, cfg.data.jitter_scale,
        cfg.data.grid_resolution, cfg.eval.surface_range
    )[0]

    # import pdb
    # pdb.set_trace()
    data['task'] = 'pick' 
    model = M2T2.from_config(cfg.m2t2)
    ckpt = torch.load(cfg.eval.checkpoint)
    model.load_state_dict(ckpt['model'])
    model = model.cuda().eval()
    inputs, xyz = data['inputs'], data['points']
    outputs = {
        'grasps': [],
        'grasp_confidence': [],
        'grasp_contacts': [],
        'placements': [],
        'placement_confidence': [],
        'placement_contacts': []
    }
    # Aggregates results (grasps)
    for _ in range(cfg.eval.num_runs):
        pt_idx = sample_points(xyz, cfg.data.num_points)
        data['inputs'] = inputs[pt_idx]
        data['points'] = xyz[pt_idx]
        # pt_idx = sample_points(obj_inputs, cfg.data.num_object_points)
        # import pdb
        # pdb.set_trace()
        data_batch = collate([data])
        to_gpu(data_batch)
        with torch.no_grad():
            model_ouputs = model.infer(data_batch, cfg.eval)
        to_cpu(model_ouputs)
        for key in outputs:
            if 'place' in key and len(outputs[key]) > 0:
                outputs[key] = [
                    torch.cat([prev, cur])
                    for prev, cur in zip(outputs[key], model_ouputs[key][0])
                ]
            else:
                outputs[key].extend(model_ouputs[key][0])
    data['inputs'], data['points'] = inputs, xyz
    return data, outputs

def load_rgb_xyz(
    meta_data, rgb, depth, seg, robot_prob, world_coord, jitter_scale, grid_res, surface_range=0
):
    normalized_rgb = normalize_rgb_batches(rgb / 255.0)
    xyz = depth_to_xyz(depth, meta_data['intrinsics']).float()
    seg = torch.from_numpy(seg).to(depth.device)
    normalized_rgb = normalized_rgb.permute(0, 2, 3, 1)
    label_map = meta_data['label_map']

    if torch.rand(()) > robot_prob:
        robot_mask = seg == label_map['robot']
        if 'robot_table' in label_map:
            robot_mask |= seg == label_map['robot_table']
        if 'object_label' in meta_data:
            robot_mask |= seg == label_map[meta_data['object_label']]
        depth[robot_mask] = 0
        seg[robot_mask] = 0
    xyz, normalized_rgb, seg = xyz[depth > 0], normalized_rgb[depth > 0], seg[depth > 0]
    cam_pose = torch.from_numpy(meta_data['camera_pose']).float().to(xyz.device)
    xyz_world = xyz @ cam_pose[:3, :3].T + cam_pose[:3, 3]

    if 'scene_bounds' in meta_data:
        bounds = meta_data['scene_bounds']
        within = (xyz_world[:, 0] > bounds[0]) & (xyz_world[:, 0] < bounds[3]) \
            & (xyz_world[:, 1] > bounds[1]) & (xyz_world[:, 1] < bounds[4]) \
            & (xyz_world[:, 2] > bounds[2]) & (xyz_world[:, 2] < bounds[5])
        xyz_world, rgb, seg = xyz_world[within], rgb[within], seg[within]
        # Set z-coordinate of all points near table to 0
        xyz_world[np.abs(xyz_world[:, 2]) < surface_range, 2] = 0
        if not world_coord:
            world2cam = cam_pose.inverse()
            xyz = xyz_world @ world2cam[:3, :3].T + world2cam[:3, 3]
    if world_coord:
        xyz = xyz_world

    if jitter_scale > 0:
        table_mask = seg == label_map['table']
        if 'robot_table' in label_map:
            table_mask |= seg == label_map['robot_table']
        xyz[table_mask] = jitter_gaussian(
            xyz[table_mask], jitter_scale, jitter_scale
        )
    outputs = {
        'inputs': torch.cat([xyz - xyz.mean(dim=0), normalized_rgb], dim=1),
        'points': xyz,
        'seg': seg,
        'cam_pose': cam_pose
    }

    if 'object_label' in meta_data:
        obj_mask = seg == label_map[meta_data['object_label']]
        obj_xyz, obj_rgb = xyz_world[obj_mask], rgb[obj_mask]
        obj_xyz_grid = torch.unique(
            (obj_xyz[:, :2] / grid_res).round(), dim=0
        ) * grid_res
        bottom_center = obj_xyz.min(dim=0)[0]
        bottom_center[:2] = obj_xyz_grid.mean(dim=0)

        ee_pose = torch.from_numpy(meta_data['ee_pose']).float()
        inv_ee_pose = ee_pose.inverse()
        obj_xyz = obj_xyz @ inv_ee_pose[:3, :3].T + inv_ee_pose[:3, 3]
        outputs.update({
            'object_inputs': torch.cat([
                obj_xyz - obj_xyz.mean(dim=0), obj_rgb
            ], dim=1),
            'ee_pose': ee_pose,
            'bottom_center': bottom_center,
            'object_center': obj_xyz.mean(dim=0)
        })
    else:
        outputs.update({
            'object_inputs': torch.rand(1024, 6),
            'ee_pose': torch.eye(4),
            'bottom_center': torch.zeros(3),
            'object_center': torch.zeros(3)
        })
    return outputs, meta_data

def visualize(cfg, data, outputs):
    vis = create_visualizer()
    rgb = denormalize_rgb(
        data['inputs'][:, 3:].T.unsqueeze(2)
    ).squeeze(2).T
    rgb = (rgb.cpu().numpy() * 255).astype('uint8')
    xyz = data['points'].cpu().numpy()
    cam_pose = data['cam_pose'].cpu().double().numpy()
    make_frame(vis, 'camera', T=cam_pose)
    if not cfg.eval.world_coord:
        xyz = xyz @ cam_pose[:3, :3].T + cam_pose[:3, 3]
    visualize_pointcloud(vis, 'scene', xyz, rgb, size=0.005)
    if data['task'] == 'pick':
        print(outputs['grasps'])
        for i, (grasps, conf, contacts, color) in enumerate(zip(
            outputs['grasps'],
            outputs['grasp_confidence'],
            outputs['grasp_contacts'],
            get_set_colors()
        )):
            print(f"object_{i:02d} has {grasps.shape[0]} grasps")
            conf = conf.numpy()
            conf_colors = (np.stack([
                1 - conf, conf, np.zeros_like(conf)
            ], axis=1) * 255).astype('uint8')
            visualize_pointcloud(
                vis, f"object_{i:02d}/contacts",
                contacts.numpy(), conf_colors, size=0.01
            )
            grasps = grasps.numpy()
            if not cfg.eval.world_coord:
                grasps = cam_pose @ grasps
            
            for j, grasp in enumerate(grasps):
                visualize_grasp(
                    vis, f"object_{i:02d}/grasps/{j:03d}",
                    grasp, color, linewidth=0.2
                )


def pos_and_quat_from_matrix(transform_mat):
    pos = transform_mat[:3, -1].clone()

    # Extract rotation matrix from the transformation matrix and convert to quaternion
    rotation_matrix = transform_mat[:3, :3]

    quat = math.quat_from_matrix(rotation_matrix)
    euler_angles = math.euler_xyz_from_quat(quat[None]) 

    # Unpack the tuple
    roll, pitch, yaw = euler_angles

    yaw -= np.pi/2 # rotate to account for rotated frame between m2t2 and isaac

    # Adjust the yaw angle
    if yaw > np.pi / 2:
        yaw -= np.pi
    if yaw < -np.pi / 2:
        yaw += np.pi

    # Convert the adjusted Euler angles back to quaternion
    adjusted_quat = math.quat_from_euler_xyz(roll, pitch, yaw).squeeze()

    return pos, adjusted_quat

def normalize_rgb_batches(rgb):
    rgb = rgb[:, :, :, :3]  
    rgb = rgb.permute(0, 3, 1, 2)  # Shape will be (batch_size, 3, height, width)
    normalize_rgb = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    normalized_rgb = normalize_rgb(rgb)
    return normalized_rgb

def depth_to_xyz(depth, intrinsics):
    if isinstance(intrinsics, np.ndarray):
        intrinsics = torch.from_numpy(intrinsics).float().to(depth.device)

    batch_size, height, width = depth.shape

    # Expand intrinsics to match the batch size
    intrinsics = intrinsics.unsqueeze(0).expand(batch_size, -1, -1)

    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    cx, cy = intrinsics[:, 0, 2], intrinsics[:, 1, 2]

    u = torch.arange(width, device=depth.device).view(1, 1, -1).expand(batch_size, height, -1)
    v = torch.arange(height, device=depth.device).view(1, -1, 1).expand(batch_size, -1, width)

    Z = depth
    X = (u - cx.view(-1, 1, 1)) * (Z / fx.view(-1, 1, 1))
    Y = (v - cy.view(-1, 1, 1)) * (Z / fy.view(-1, 1, 1))

    xyz = torch.stack((X, Y, Z), dim=-1)
    return xyz
