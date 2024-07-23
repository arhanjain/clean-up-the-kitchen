import math
from PIL import Image
from sklearn.neighbors import KDTree
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import torch

from M2T2.m2t2.dataset_utils import sample_points, depth_to_xyz, normalize_rgb
from M2T2.m2t2.dataset import collate
from M2T2.m2t2.m2t2 import M2T2
from M2T2.m2t2.plot_utils import get_set_colors
from M2T2.m2t2.train_utils import to_cpu, to_gpu
from M2T2.m2t2.dataset_utils import denormalize_rgb, sample_points, jitter_gaussian
from M2T2.m2t2.meshcat_utils import (
    create_visualizer, make_frame, visualize_grasp, visualize_pointcloud
)


def load_and_predict(cfg, meta_data, rgb, depth, seg):
    data =  load_rgb_xyz(
        meta_data, rgb, depth, seg, cfg.data.robot_prob,
        cfg.data.world_coord, cfg.data.jitter_scale,
        cfg.data.grid_resolution, cfg.eval.surface_range
    )[0]
    # data returns a tuple for each env

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
    depth = depth[0].cpu()
    normalized_rgb = normalize_rgb(rgb[0][:, :, :3].float().cpu().numpy() / 255.0)
    xyz = torch.from_numpy(
        depth_to_xyz(depth.cpu().numpy(), meta_data['intrinsics'])
    ).float()
    seg = torch.from_numpy(seg)
    normalized_rgb = normalized_rgb.permute(1, 2, 0)
    label_map = meta_data['label_map']

    if torch.rand(()) > robot_prob:
        robot_mask = seg == label_map['robot']
        if 'robot_table' in label_map:
            robot_mask |= seg == label_map['robot_table']
        if 'object_label' in meta_data:
            robot_mask |= seg == label_map[meta_data['object_label']]
        depth[robot_mask] = 0
        seg[robot_mask] = 0
    xyz, normalized_rgb, seg = xyz[depth > 0], normalized_rgb[depth > 0], seg[0][depth > 0]
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
# def load_rgb_xyz(meta_data, rgb, depth, seg):
#     depth = depth[0]
#     normalized_rgb = normalize_rgb(rgb[0][:, :, :3].float().cpu().numpy() / 255.0)
#     xyz = torch.from_numpy(
#         depth_to_xyz(depth, meta_data['intrinsics'])
#     ).float()
#     seg = torch.from_numpy(np.array(seg.cpu()))
#     normalized_rgb = normalized_rgb.permute(1, 2, 0)
#     # Filter out points where depth is zero
#     # Turns it into #307200, 3
#     xyz, normalized_rgb, seg = xyz[depth > 0], normalized_rgb[depth > 0], seg[0][depth > 0]

#     outputs = {
#         'inputs': torch.cat([xyz - xyz.mean(dim=0), normalized_rgb], dim=1),
#         'points': xyz,
#         'seg': seg,
#     }

#     if 'camera_pose' in meta_data:
#         cam_pose = torch.from_numpy(meta_data['camera_pose']).float()
#         outputs['cam_pose'] = cam_pose

#     outputs.update({
#         'object_inputs': torch.rand(1024, 6),
#         'ee_pose': torch.eye(4),
#         'bottom_center': torch.zeros(3),
#         'object_center': torch.zeros(3)
#     })

#     return outputs, meta_data

def visualize(cfg, data, outputs):
    vis = create_visualizer()
    rgb = denormalize_rgb(
        data['inputs'][:, 3:].T.unsqueeze(2)
    ).squeeze(2).T
    rgb = (rgb.numpy() * 255).astype('uint8')
    xyz = data['points'].numpy()
    cam_pose = data['cam_pose'].double().numpy()
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
    # elif data['task'] == 'place':
    #     ee_pose = data['ee_pose'].double().numpy()
    #     make_frame(vis, 'ee', T=ee_pose)
    #     obj_xyz_ee, obj_rgb = data['object_inputs'].split([3, 3], dim=1)
    #     obj_xyz_ee = (obj_xyz_ee + data['object_center']).numpy()
    #     obj_xyz = obj_xyz_ee @ ee_pose[:3, :3].T + ee_pose[:3, 3]
    #     obj_rgb = denormalize_rgb(obj_rgb.T.unsqueeze(2)).squeeze(2).T
    #     obj_rgb = (obj_rgb.numpy() * 255).astype('uint8')
    #     visualize_pointcloud(vis, 'object', obj_xyz, obj_rgb, size=0.005)
    #     for i, (placements, conf, contacts) in enumerate(zip(
    #         outputs['placements'],
    #         outputs['placement_confidence'],
    #         outputs['placement_contacts'],
    #     )):
    #         print(f"orientation_{i:02d} has {placements.shape[0]} placements")
    #         conf = conf.numpy()
    #         conf_colors = (np.stack([
    #             1 - conf, conf, np.zeros_like(conf)
    #         ], axis=1) * 255).astype('uint8')
    #         visualize_pointcloud(
    #             vis, f"orientation_{i:02d}/contacts",
    #             contacts.numpy(), conf_colors, size=0.01
    #         )
    #         placements = placements.numpy()
    #         if not cfg.eval.world_coord:
    #             placements = cam_pose @ placements
    #         visited = np.zeros((0, 3))
    #         for j, k in enumerate(np.random.permutation(placements.shape[0])):
    #             if visited.shape[0] > 0:
    #                 dist = np.sqrt((
    #                     (placements[k, :3, 3] - visited) ** 2
    #                 ).sum(axis=1))
    #                 if dist.min() < cfg.eval.placement_vis_radius:
    #                     continue
    #             visited = np.concatenate([visited, placements[k:k+1, :3, 3]])
    #             visualize_grasp(
    #                 vis, f"orientation_{i:02d}/placements/{j:02d}/gripper",
    #                 placements[k], [0, 255, 0], linewidth=0.2
    #             )
    #             obj_xyz_placed = obj_xyz_ee @ placements[k, :3, :3].T \
    #                            + placements[k, :3, 3]
    #             visualize_pointcloud(
    #                 vis, f"orientation_{i:02d}/placements/{j:02d}/object",
    #                 obj_xyz_placed, obj_rgb, size=0.01
    #             )

# def get_grasping_points(cfg, meta_data, rgb, depth, seg):
#     print('depth', depth)
#     data, outputs = load_and_predict(cfg, meta_data, rgb, depth, seg)
#     # vis = create_visualizer()
#     # rgb = denormalize_rgb(
#     #     data['inputs'][:, 3:].T.unsqueeze(2)
#     # ).squeeze(2).T
#     # rgb = (rgb.numpy()).astype('uint8')
#     # xyz = data['points'].numpy()
#     # cam_pose = data['cam_pose'].double().numpy()
#     # make_frame(vis, 'camera', T=cam_pose)
#     # if not cfg.eval.world_coord:
#     #     xyz = xyz @ cam_pose[:3, :3].T + cam_pose[:3, 3]
#     # visualize_pointcloud(vis, 'scene', xyz, rgb, size=0.005)
#     # if data['task'] == 'pick':
#     #     for i, (grasps, conf, contacts, color) in enumerate(zip(
#     #         outputs['grasps'],
#     #         outputs['grasp_confidence'],
#     #         outputs['grasp_contacts'],
#     #         get_set_colors()
#     #     )):
#     #         print(f"object_{i:02d} has {grasps.shape[0]} grasps")
#     #         conf = conf.numpy()
#     #         conf_colors = (np.stack([
#     #             1 - conf, conf, np.zeros_like(conf)
#     #         ], axis=1) * 255).astype('uint8')
#     #         visualize_pointcloud(
#     #             vis, f"object_{i:02d}/contacts",
#     #             contacts.numpy(), conf_colors, size=0.01
#     #         )
#     #         grasps = grasps.cpu().numpy()
#     #         if not cfg.eval.world_coord:
#     #             grasps = cam_pose @ grasps
            
#     #         for j, grasp in enumerate(grasps):
#     #             visualize_grasp(
#     #                 vis, f"object_{i:02d}/grasps/{j:03d}",
#     #                 grasp, color, linewidth=0.2
#     #             )
#     if data['task'] == 'pick':
#         for i, (grasps, conf) in enumerate(zip(
#             outputs['grasps'],
#             outputs['grasp_confidence']
#         )):
#             print(f"object_{i:02d} has {grasps.shape[0]} grasps")
#             grasps = torch.tensor(grasps, dtype = torch.float32)
#             conf = torch.tensor(conf)
#             conf_highest_index = torch.argmax(conf)
#             best_grasp = grasps[conf_highest_index]
#     return grasps



    