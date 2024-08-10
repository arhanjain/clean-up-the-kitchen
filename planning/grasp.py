import math
from PIL import Image
from sklearn.neighbors import KDTree
from torch.utils.data import Dataset
import numpy as np
import os
import re
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
from pxr import Usd
import omni.isaac.lab.utils.math as math

class Grasper:
    '''
    Predicts grasps and placement poses for objects in the environment.
    Supports simulated and real data. 

    Parameters
    ----------
    grasp_cfg: OmegaConf
        The configuration for the grasp prediction system.
    usd_path: str
        The path to the USD file containing the environment.
    device: int
        The device to run the model on. Preferable same GPU as the environment.

    '''
    def __init__(self, grasp_cfg, usd_path=None, device=0) -> None:
        # Load Model
        self._model = M2T2.from_config(grasp_cfg.m2t2)
        ckpt = torch.load(grasp_cfg.eval.checkpoint)
        self._model.load_state_dict(ckpt["model"])
        self._model = self._model.to(device).eval()

        self._usd_path = usd_path
        self._synthetic_pcd = grasp_cfg.data.synthetic_pcd
        self._cfg = grasp_cfg

    def get_placement(self, env, object_class, manipulation_type="placements", viz=True):
        self.get_manipulation(self, env, object_class, manipulation_type, viz=True)

    def get_grasp(self, env, object_class, manipulation_type="grasps", viz=True):
        self.get_manipulation(self, env, object_class, manipulation_type, viz=True)
        
    
    def get_manipulation(self, env, object_class, manipulation_type, viz=True):
        '''
        Returns the best grasp or place pose on the specified object for each environment.
        Given N environments, only M may be successful. The returned pose tensor
        will be of shape M.

        Parameters
        ----------
        env: ManagerBasedRLEnv (with possible wrappers)
            The environment.
        object_class: str
            The class of object to interact with, used in semantic segmentation.
        manipulation_type: str
            The type of manipulation to perform, either 'grasps' or 'placements'.
        viz : bool, optional, default: True
            Whether to visualize the scene.

        Returns
        -------
        pose: torch.tensor(float) shape: (M, 7)
            The position and quaternion of the best pose (grasp or place) for each environment.
        success: torch.tensor(bool) shape: (N,)
            Whether the manipulation prediction was successful for each environment.
        '''
        
        goal_pos, goal_quat, success = None, None, None
        # Obtains camera-observed PCDs and selects the highest confidence manipulation per environment
        if self._synthetic_pcd:
            # Samples synthetic PCD points from the mesh and selects the best manipulation (grasp or place) 
            # from the same set of predicted poses (more efficient!)
            data, outputs = self.load_and_predict_synthetic(manipulation_type)
            (goal_pos, goal_quat), success = self.sample_manipulations(outputs, env.unwrapped.num_envs, manipulation_type)
        else:
            rgb, seg, depth, meta_data = env.get_camera_data()
            data = rgb, seg, depth, meta_data
            data, outputs = self.load_and_predict_real(data, self._model, self._cfg, manipulation_type, obj_label=object_class)
            (goal_pos, goal_quat), success = self.choose_manipulation(outputs, manipulation_type)

        if viz:
            self.visualize(data[0], {k: v[0] for k, v in outputs.items()})

        if not torch.all(success):
            return (None, None), success

        return torch.cat([goal_pos, goal_quat], dim=1), success

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

        pregrasp = grasp_pose.clone()
        pregrasp[:, :3] -= offset * direction

        return pregrasp

    def sample_manipulations(self, outputs, num_manipulations, manipulation_type):
        '''
        Weighted samples `num_manipulations` from the outputs of `load_and_predict`. Returns the
        sampled manipulations (either grasps or placements) in the form of:
        - pos: (num_manipulations, 3)
        - quat: (num_manipulations, 4)

        Parameters
        ----------
        outputs: dict
            The outputs from `load_and_predict`, containing manipulation data and confidence scores.
        num_manipulations: int
            The number of manipulations (grasps or placements) to sample.
        manipulation_type: str
            The type of manipulation to perform, either 'grasps' or 'placements'.

        Returns
        -------
        pos: torch.tensor(float) shape: (num_manipulations, 3)
            The positions of the sampled manipulations.
        quat: torch.tensor(float) shape: (num_manipulations, 4)
            The quaternions of the sampled manipulations.
        success: torch.tensor(bool) shape: (num_manipulations,)
            A tensor indicating successful manipulation predictions for each sampled manipulation.
        '''
        if manipulation_type == 'grasps':
            all_manipulations = outputs["grasps"][0]
            all_conf = outputs["grasp_confidence"][0]
        else:
            all_manipulations = outputs['placements'][0]
            all_conf = outputs["placement_confidence"][0]
        if len(all_manipulations) == 0:
            return (None, None), torch.zeros(num_manipulations, dtype=torch.bool)
        all_manipulations = torch.concatenate(all_manipulations, dim=0)
        all_conf = torch.concatenate(all_conf, dim=0)

        prob_dist = torch.softmax(all_conf, dim=0)
        indices = np.random.choice(all_manipulations.shape[0], num_manipulations, replace=True, p=prob_dist.numpy())

        selected_manipulations = all_manipulations[indices]
        pos, quat = self.m2t2_grasp_to_pos_and_quat(selected_manipulations)
        return (pos, quat), torch.ones(num_manipulations, dtype=torch.bool)
        
    def load_and_predict_synthetic(self, manipulation_type):
        '''
        Load and prepare synthetic point clouds and predict manipulations (either grasps or placements).
        '''
        usd_stage = Usd.Stage.Open(self.usd_path)
        geometries = []
        pattern = re.compile(r'/World/Xform_.*/Object_Geometry$')
        for prim in usd_stage.TraverseAll():
            path = str(prim.GetPath())
            if re.match(pattern, path):
                points = np.array(prim.GetAttribute("points").Get())
                points_homogeneous = np.concatenate((points, np.ones((points.shape[0],1))), axis=1)
                points_homogeneous = torch.tensor(points_homogeneous)
                transformation = torch.tensor(prim.GetParent().GetParent().GetAttribute("xformOp:transform").Get())
                rot = transformation[:3, :3]
                pos = torch.tensor([transformation[-1, 0], -transformation[-1,2], transformation[-1,1], 1])
                transformation = torch.eye(4, dtype=torch.float64)
                transformation[:3,:3] = rot
                transformation[:, -1] = pos

                new_points_homo = points_homogeneous @ transformation.T

                new_points = new_points_homo[:, :3] / new_points_homo[:, 3:]

                geometries.append(new_points)

        all_points = np.concatenate(geometries, axis=0)
        seg = np.ones(all_points.shape[0])

        # Make M2T2 predictions             
        xyz = torch.tensor(all_points, dtype=torch.float32)
        seg = torch.tensor(seg, dtype=torch.float32)
        # cam_pose = torch.tensor(metadata[0]["camera_pose"])
        rgb = torch.ones_like(xyz)
        d = {
            'inputs': torch.cat([xyz - xyz.mean(dim=0), rgb], dim=1),
            'points': xyz,
            'seg': seg,
            'cam_pose': torch.eye(4),
        }
        d.update({
            'object_inputs': torch.rand(1024, 6),
            'ee_pose': torch.eye(4),
            'bottom_center': torch.zeros(3),
            'object_center': torch.zeros(3)
        })

        if manipulation_type == 'grasps':
            d['task'] = 'pick'
        else:
            d['task'] = 'place'

        inputs, xyz, seg = d['inputs'], d['points'], d['seg']
        obj_inputs = d['object_inputs']

        pt_idx = sample_points(xyz, 25_000)
        d['inputs'] = inputs[pt_idx]
        d['points'] = xyz[pt_idx]
        d['seg'] = seg[pt_idx]
        pt_idx = sample_points(obj_inputs, self._cfg.data.num_sample_obj_points)
        d['object_inputs'] = obj_inputs[pt_idx]

        data = [d]
        outputs = {
            'grasps': [],
            'grasp_confidence': [],
            'grasp_contacts': [],
            'placements': [],
            'placement_confidence': [],
            'placement_contacts': []
        }

        data_batch = collate(data)
        to_gpu(data_batch)

        with torch.no_grad():
            model_outputs = self.model.infer(data_batch, self._cfg.eval)
        to_cpu(model_outputs)
        for key in outputs:
            if 'placements' in key and len(outputs[key]) > 0:
                outputs[key] = [
                    torch.cat([prev, cur])
                    for prev, cur in zip(outputs[key], model_outputs[key][0])
                ]
            else:
                outputs[key].extend(model_outputs[key])

        return data, outputs

    @staticmethod
    def load_rgb_xyz(
        loaded_data, robot_prob, world_coord, jitter_scale, grid_res, surface_range=0
    ):
        '''
        Loads and processes RGB-D data into point clouds, considering segmentation masks and potential 
        noise or transformations to improve generalization.
        
        Parameters
        ----------
        loaded_data: tuple
            The tuple containing RGB, segmentation, depth images, and metadata.
        robot_prob: float
            The probability of masking out the robot's points from the point cloud.
        world_coord: bool
            Whether to return the point cloud in world coordinates or camera coordinates.
        jitter_scale: float
            The scale of the jitter noise to be applied to the point cloud.
        grid_res: float
            The resolution of the grid used to downsample the object point cloud.
        surface_range: float
            The range within which points are considered to be on a surface (e.g., table).

        Returns
        -------
        outputs: dict
            A dictionary containing processed inputs, point clouds, segmentation masks, and camera poses.
        meta_data: dict
            Metadata associated with the input data.
        '''
        rgb, seg, depth, meta_data = loaded_data
        rgb = normalize_rgb(rgb).permute(1, 2, 0)
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

    def load_and_predict_real(self, loaded_data, model, cfg, manipulation_type, obj_label=None):
        '''
        Processes real-world RGB-D data to generate manipulation predictions (grasps or placements).
        
        Parameters
        ----------
        loaded_data: tuple
            The tuple containing RGB, segmentation, depth images, and metadata.
        model: M2T2
            The manipulation prediction model (e.g., M2T2).
        cfg: OmegaConf
            The configuration for the manipulation prediction system.
        obj_label: str, optional
            The label of the object to focus on in the scene.
        manipulation_type: str, optional
            The type of manipulation to predict, either 'pick' (for grasps) or 'place'.

        Returns
        -------
        data: list
            A list of dictionaries containing processed point cloud data and additional information.
        outputs: dict
            A dictionary containing the predicted grasps or placements and their confidence scores.
        '''
        data, meta_data = [], []
        batch_size = loaded_data[0].shape[0]
        for i in range(batch_size):
            batch_element = [elem[i] for elem in loaded_data]
            if obj_label:  # add object label to metadata
                batch_element[-1]["object_label"] = obj_label
            d, meta = self.load_rgb_xyz(
                batch_element, cfg.data.robot_prob,
                cfg.data.world_coord, cfg.data.jitter_scale,
                cfg.data.grid_resolution, cfg.eval.surface_range
            )
            # Set the task type based on manipulation_type
            if manipulation_type == 'grasps':
                d['task'] = 'pick'
            else:
                d['task'] = 'place'

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

        outputs = {
            'grasps': [],
            'grasp_confidence': [],
            'grasp_contacts': [],
            'placements': [],
            'placement_confidence': [],
            'placement_contacts': []
        }

        data_batch = collate(data)
        to_gpu(data_batch)

        with torch.no_grad():
            model_outputs = model.infer(data_batch, cfg.eval)
        to_cpu(model_outputs)
        for key in outputs:
            if 'place' in key and len(outputs[key]) > 0:
                outputs[key] = [
                    torch.cat([prev, cur])
                    for prev, cur in zip(outputs[key], model_outputs[key][0])
                ]
            else:
                outputs[key].extend(model_outputs[key])

        return data, outputs

    def choose_manipulation(self, outputs, manipulation_type):
        '''
        Takes in the output of load_and_predict. Input will have N environments, out 
        of which M will be successes. Outputs the best manipulation per environment
        in the form of pos: (M, 3), quat: (M, 4). Also reports whether manipulation prediction
        was successful (length N).

        Parameters
        ----------
        outputs: dict
            The dictionary containing manipulation data and confidence scores.
        manipulation_type: str
            Specifies whether to choose grasps or placements.

        Returns
        -------
        pos: torch.tensor(float) shape: (M, 3)
            The positions of the best manipulations.
        quat: torch.tensor(float) shape: (M, 4)
            The quaternions of the best manipulations.
        success: torch.tensor(bool) shape: (N,)
            A tensor indicating whether manipulation predictions were successful.
        '''
        best_manipulations = []
        successes = []
        
        # Determine the correct confidence key based on manipulation_type
        if manipulation_type == "grasps":
            conf_key = "grasp_confidence"
        else:
            conf_key = "placement_confidence"
        
        for i in range(len(outputs[manipulation_type])):  # iterates over the number of environments
            if len(outputs[manipulation_type][i]) == 0:
                successes.append(False)
                continue
            manipulations = np.concatenate(outputs[manipulation_type][i], axis=0)
            manipulation_conf = np.concatenate(outputs[conf_key][i], axis=0)
            sorted_manipulation_idxs = np.argsort(manipulation_conf, axis=0)  # ascending order of confidence
            manipulations = manipulations[sorted_manipulation_idxs]
            best_manipulations.append(manipulations[-1])
            successes.append(True)

        # Convert manipulation poses from M2T2 form to Isaac form
        best_manipulations = torch.tensor(best_manipulations)
        if len(best_manipulations) == 0:
            pos, quat = torch.zeros(1, 1), torch.zeros(1, 1)
        else:
            pos, quat = self.m2t2_grasp_to_pos_and_quat(best_manipulations)
        
        return (pos, quat), torch.tensor(successes)

    
    @staticmethod
    def m2t2_grasp_to_pos_and_quat(transform_mat):
        '''
        Converts a transformation matrix from M2T2 format to position and quaternion format.

        Parameters
        ----------
        transform_mat: torch.tensor(float) shape: (N, 4, 4)
            The transformation matrix from M2T2.

        Returns
        -------
        pos: torch.tensor(float) shape: (N, 3)
            The positions extracted from the transformation matrix.
        quat: torch.tensor(float) shape: (N, 4)
            The quaternions extracted from the transformation matrix.
        '''
        pos = transform_mat[..., :3, -1].clone()

        # Extract rotation matrix from the transformation matrix and convert to quaternion
        rotation_matrix = transform_mat[..., :3, :3]

        quat = math.quat_from_matrix(rotation_matrix)
        euler_angles = math.euler_xyz_from_quat(quat) 

        # Unpack the tuple
        roll, pitch, yaw = euler_angles

        yaw -= np.pi/2  # rotate to account for rotated frame between M2T2 and Isaac

        # Adjust the yaw angle
        yaw = torch.where(yaw > np.pi/2, yaw - np.pi, yaw)
        yaw = torch.where(yaw > np.pi/2, yaw + np.pi, yaw)

        # Convert the adjusted Euler angles back to quaternion
        adjusted_quat = math.quat_from_euler_xyz(roll, pitch, yaw).view(-1, 4)

        return pos, adjusted_quat

    def visualize(self, data, outputs):
        '''
        Visualizes the point clouds and manipulation predictions.

        Parameters
        ----------
        data: dict
            The processed input data including point clouds and segmentation.
        outputs: dict
            The dictionary containing manipulation predictions.
        '''
        vis = create_visualizer()
        rgb = denormalize_rgb(
            data['inputs'][:, 3:].T.unsqueeze(2)
        ).squeeze(2).T
        rgb = (rgb.cpu().numpy() * 255).astype('uint8')
        xyz = data['points'].cpu().numpy()
        cam_pose = data['cam_pose'].cpu().double().numpy()
        make_frame(vis, 'camera', T=cam_pose)
        if not self._cfg.eval.world_coord:
            xyz = xyz @ cam_pose[:3, :3].T + cam_pose[:3, 3]
        visualize_pointcloud(vis, 'scene', xyz, rgb, size=0.005)
        if data['task'] == 'pick':
            print("Visualizing picking")
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
                if not self._cfg.eval.world_coord:
                    grasps = cam_pose @ grasps
                
                for j, grasp in enumerate(grasps):
                    visualize_grasp(
                        vis, f"object_{i:02d}/grasps/{j:03d}",
                        grasp, color, linewidth=0.2
                    )
        elif data['task'] == 'place':
            print("Visualizing placement")
            ee_pose = data['ee_pose'].double().numpy()
            make_frame(vis, 'ee', T=ee_pose)
            obj_xyz_ee, obj_rgb = data['object_inputs'].split([3, 3], dim=1)
            obj_xyz_ee = (obj_xyz_ee + data['object_center']).numpy()
            obj_xyz = obj_xyz_ee @ ee_pose[:3, :3].T + ee_pose[:3, 3]
            obj_rgb = denormalize_rgb(obj_rgb.T.unsqueeze(2)).squeeze(2).T
            obj_rgb = (obj_rgb.numpy() * 255).astype('uint8')
            visualize_pointcloud(vis, 'object', obj_xyz, obj_rgb, size=0.005)
            for i, (placements, conf, contacts) in enumerate(zip(
                outputs['placements'],
                outputs['placement_confidence'],
                outputs['placement_contacts'],
            )):
                print(f"orientation_{i:02d} has {placements.shape[0]} placements")
                conf = conf.numpy()
                conf_colors = (np.stack([
                    1 - conf, conf, np.zeros_like(conf)
                ], axis=1) * 255).astype('uint8')
                visualize_pointcloud(
                    vis, f"orientation_{i:02d}/contacts",
                    contacts.numpy(), conf_colors, size=0.01
                )
                placements = placements.numpy()
                if not self._cfg.eval.world_coord:
                    placements = cam_pose @ placements
                visited = np.zeros((0, 3))
                for j, k in enumerate(np.random.permutation(placements.shape[0])):
                    if visited.shape[0] > 0:
                        dist = np.sqrt((
                            (placements[k, :3, 3] - visited) ** 2
                        ).sum(axis=1))
                        if dist.min() < self.cfg.eval.placement_vis_radius:
                            continue
                    visited = np.concatenate([visited, placements[k:k+1, :3, 3]])
                    visualize_grasp(
                        vis, f"orientation_{i:02d}/placements/{j:02d}/gripper",
                        placements[k], [0, 255, 0], linewidth=0.2
                    )
                    obj_xyz_placed = obj_xyz_ee @ placements[k, :3, :3].T \
                                + placements[k, :3, 3]
                    visualize_pointcloud(
                        vis, f"orientation_{i:02d}/placements/{j:02d}/object",
                        obj_xyz_placed, obj_rgb, size=0.01
                    )