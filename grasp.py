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

class Grasper:
    def __init__(self, model, cfg) -> None:
        self.model = model
        self.cfg = cfg

    def get_grasp(self, data, object_class, viz=True):
        '''
        Returns best grasp pose on specified object for each environment.
        Given N environments, only M may be successful. Returned grasp tensor
        will be of shape M.

        Parameters
        ----------
        data: tuple
            The tuple contains (rgb, seg, depth, meta_data) batched for each environment
        object_class: str
            The class of object to grasp, used in semantic segmentation
        viz: bool
            Whether to visualize the scene

        Returns
        -------
        pose: torch.tensor(float) shape: (M, 7)
            The position and quaternion of the best grasp pose for each environment
        success: torch.tensor(bool) shape: (N,)
            Whether the grasp prediction was successful for each environment
        '''
        

        data, outputs = load_and_predict(data, self.model, self.cfg, obj_label=object_class)
        if viz:
            visualize(self.cfg, data[0], {k: v[0] for k, v in outputs.items()})
        (goal_pos, goal_quat), success = choose_grasp(outputs)

        return torch.cat([goal_pos,goal_quat], dim=1), torch.tensor(success)

    @staticmethod
    def get_pregrasp(grasp_pose, offset):
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

        pos, quat = grasp_pose[:, :3], grasp_pose[:, 3:]

        # Normalize quaternions
        norms = torch.norm(quat, dim=1, keepdim=True)
        q = quat / norms

        # Calculate direction vectors
        direction = torch.empty((quat.shape[0], 3), device=quat.device)
        direction[:, 0] = 2 * (q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3])
        direction[:, 1] = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
        direction[:, 2] = 1 - 2 * (q[:, 0]**2 + q[:, 1]**2)

        pregrasp = grasp_pose.clone()
        pregrasp[:, :3] -= offset * direction

        return pregrasp


def load_rgb_xyz(
    loaded_data, robot_prob, world_coord, jitter_scale, grid_res, surface_range=0
):
    rgb, seg, depth, meta_data = loaded_data
    rgb = normalize_rgb(rgb).permute(1,2,0)
    xyz = torch.from_numpy(
        depth_to_xyz(depth, meta_data['intrinsics'])
    ).float()
    seg = torch.from_numpy(np.array(seg))

    label_map = meta_data['label_map']

    if torch.rand(()) > robot_prob:
        robot_mask = seg == label_map['robot']
        if 'robot_table' in label_map:
            robot_mask |= seg == label_map['robot_table']
        if 'object_label' in meta_data:
            robot_mask |= seg == label_map[meta_data['object_label']]
        depth[robot_mask] = 0
        seg[robot_mask] = 0
    xyz, rgb, seg = xyz[depth > 0], rgb[depth > 0], seg[depth > 0]
    cam_pose = torch.from_numpy(meta_data['camera_pose']).float()
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
        'inputs': torch.cat([xyz - xyz.mean(dim=0), rgb], dim=1),
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


def load_and_predict(loaded_data, model, cfg, obj_label = None):
    data, meta_data= [], []
    batch_size = loaded_data[0].shape[0]
    for i in range(batch_size):
        batch_element = [elem[i] for elem in loaded_data]
        if obj_label: # add object label to metadata
            batch_element[-1]["object_label"] = obj_label
        d, meta = load_rgb_xyz(
            batch_element, cfg.data.robot_prob,
            cfg.data.world_coord, cfg.data.jitter_scale,
            cfg.data.grid_resolution, cfg.eval.surface_range
        )

        if 'object_label' in d:
            d['task'] = 'place'
        else:
            d['task'] = 'pick'

        inputs, xyz, seg = d['inputs'], d['points'], d['seg']
        obj_inputs = d['object_inputs']

        pt_idx = sample_points(xyz, cfg.data.num_points)
        d['inputs'] = inputs[pt_idx]
        d['points'] = xyz[pt_idx]
        d['seg'] = seg[pt_idx]
        pt_idx = sample_points(obj_inputs, cfg.data.num_object_points)
        d['object_inputs'] = obj_inputs[pt_idx]

        data.append(d)
        meta_data.append(meta)


    # model = M2T2.from_config(cfg.m2t2)
    # ckpt = torch.load(cfg.eval.checkpoint)
    # model.load_state_dict(ckpt['model'])
    # model = model.cuda().eval()

    outputs = {
        'grasps': [],
        'grasp_confidence': [],
        'grasp_contacts': [],
        'placements': [],
        'placement_confidence': [],
        'placement_contacts': []
    }
    # for _ in range(cfg.eval.num_runs):
    data_batch = collate(data)
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
            outputs[key].extend(model_ouputs[key])

    # data['inputs'], data['points'], data['seg'] = inputs, xyz, seg
    # data['object_inputs'] = obj_inputs
    return data, outputs

def choose_grasp(outputs):
    '''
    Takes in the output of load_and_predict. Input will have N environments, out 
    of which M will be successes. Outputs the best grasp per environment
    in the form of pos: (M, 3), quat: (M, 4). Also reports whether grasp prediction
    was successful (length N)

    Returns: (pos, quat), success: np.array(bool)
    '''
    best_grasps = []
    successes = []
    for i in range(len(outputs["grasps"])): # iterates num envs
        if len(outputs["grasps"][i]) == 0:
            successes.append(False)
            continue
        grasps = np.concatenate(outputs["grasps"][i], axis=0)
        grasp_conf = np.concatenate(outputs["grasp_confidence"][i], axis=0)
        sorted_grasp_idxs = np.argsort(grasp_conf, axis=0) # ascending order of confidence
        grasps = grasps[sorted_grasp_idxs]
        best_grasps.append(grasps[-1])
        successes.append(True)
    
    # Turn grasp poses from M2T2 form to Isaac form
    best_grasps = torch.tensor(best_grasps)
    if len(best_grasps) == 0:
        pos, quat = torch.zeros(1,1), torch.zeros(1,1)
    else:
        pos, quat = m2t2_grasp_to_pos_and_quat(best_grasps)
    return (pos, quat), np.array(successes)


def m2t2_grasp_to_pos_and_quat(transform_mat):
    pos = transform_mat[..., :3, -1].clone()

    # Extract rotation matrix from the transformation matrix and convert to quaternion
    rotation_matrix = transform_mat[..., :3, :3]

    quat = math.quat_from_matrix(rotation_matrix)
    euler_angles = math.euler_xyz_from_quat(quat) 

    # Unpack the tuple
    roll, pitch, yaw = euler_angles

    yaw -= np.pi/2 # rotate to account for rotated frame between m2t2 and isaac

    # Adjust the yaw angle
    yaw = torch.where(yaw > np.pi/2, yaw - np.pi, yaw)
    yaw = torch.where(yaw > np.pi/2, yaw + np.pi, yaw)

    # Convert the adjusted Euler angles back to quaternion
    adjusted_quat = math.quat_from_euler_xyz(roll, pitch, yaw).view(-1, 4)

    return pos, adjusted_quat

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

