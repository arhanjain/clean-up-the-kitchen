import open3d
import yaml
import re
import torch
import numpy as np
import argparse

from pxr import Usd
from omni.isaac.lab.utils import math

def matrix_to_pos_and_quat(matrix: torch.Tensor):
    pos = matrix[-1, :3]
    quat = math.quat_from_matrix(matrix[:3, :3])
    return pos, quat

def show_pcd(prim):
    pcd = []
    pattern = re.compile(r'/World/Xform_.*/Object_Geometry$')
    for child in Usd.PrimRange(prim):
        path = child.GetPath().pathString
        if pattern.match(path):
            points = np.array(child.GetAttribute("points").Get())
            points_homogeneous = np.concatenate((points, np.ones((points.shape[0],1))), axis=1)
            points_homogeneous = torch.tensor(points_homogeneous)
            transformation = torch.tensor(child.GetParent().GetParent().GetAttribute("xformOp:transform").Get())
            rot = transformation[:3, :3]
            pos = torch.tensor([transformation[-1, 0], -transformation[-1,2], transformation[-1,1], 1])
            transformation = torch.eye(4, dtype=torch.float64)
            transformation[:3,:3] = rot
            transformation[:, -1] = pos

            new_points_homo = points_homogeneous @ transformation.T

            new_points = new_points_homo[:, :3] / new_points_homo[:, 3:]

            pcd.append(new_points)

    pcd = np.concatenate(pcd, axis=0)

    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(pcd)
    open3d.visualization.draw_geometries([cloud])
    res = input("Object Name? Disable gravity? - Answer in format [Name],[yes/no]: ")
    name, disable_gravity = res.split(",")
    disable_gravity = True if disable_gravity == "yes" else False
    return name, disable_gravity

def get_pos_quat(prim):
    transform = prim.GetChildren()[-1].GetAttribute("xformOp:transform").Get()
    transform = torch.tensor(transform)
    pos, quat = matrix_to_pos_and_quat(transform)
    return pos.tolist(), quat.tolist()

    

def process_site(prim):
    transform = prim.GetChildren()[0].GetAttribute("xformOp:transform").Get()
    transform = torch.tensor(transform)
    pos, _ = matrix_to_pos_and_quat(transform)
    pos = pos.tolist()
    source = prim.GetChildren()[0].GetChild("FixedJoint").GetRelationship("physics:body0").GetTargets()[0].pathString.split("/")[2]
    return source, pos


# usd_path = "/home/arhan/Downloads/scene.usdz"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd_path", type=str)
    args  = parser.parse_args()

    stage = Usd.Stage.Open(args.usd_path)

    mapping = {
        "xforms": {},
        "sites": {},
        "usd_path": args.usd_path
    }
    rev_map = {}

    # Get all xforms
    pattern = re.compile(r'/World/Xform_.*$')
    for prim in stage.GetDefaultPrim().GetChildren():
        path = prim.GetPath().pathString
        if pattern.match(path):
            name, disable_gravity = show_pcd(prim)
            pos, quat = get_pos_quat(prim)
            mapping["xforms"][name] = {
                "subpath": path,
                "position": pos,
                "quaternion": quat,
                "disable_gravity": disable_gravity
            }
            rev_map[path.split("/")[-1]] = name

    # Get all sites (source xforms and transforms)
    site_pattern = re.compile(r'^/World/SiteXform_\d+$')
    for prim in stage.GetDefaultPrim().GetChildren():
        path = prim.GetPath().pathString
        if site_pattern.match(path):
            source_name, position = process_site(prim)
            site_name = f"{rev_map[source_name]}_site"
            mapping["sites"][site_name] = {
                "source": rev_map[source_name],
                "position": position
            }


    usd_name = args.usd_path.split("/")[-1].split(".")[0]
    with open(f"./data/{usd_name}_xform_mapping.yml", "w") as f:
        yaml.dump(mapping, f)
