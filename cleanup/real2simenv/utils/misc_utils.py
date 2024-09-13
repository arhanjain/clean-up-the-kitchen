import torch
import numpy as np
import omni.isaac.lab.utils.math as math

def pos_and_quat_to_matrix(pos, quat):
    '''
    Returns the 4x4 transformation matrix derived from the
    position and quaternion
    '''

    num_envs = pos.shape[0]

    rot_mat = math.matrix_from_quat(quat)
    pos = pos.view(-1, 3, 1)

    transformation = torch.cat((rot_mat, pos), dim=2).cpu()

    bottom_row = torch.tensor([0, 0, 0, 1]).expand(num_envs, 1, 4)
    transformation = torch.cat((transformation, bottom_row), dim=1).numpy()

    return transformation

def GUI_matrix_to_pos_and_quat(matrix: torch.Tensor):
    '''
    Takes in a 4x4 matrix,
    '''
    pos = matrix[-1, :3]
    temp = pos.clone()
    pos[2] = temp[1]
    pos[1] = -temp[2]
    quat = math.quat_from_matrix(matrix[:3, :3])
    return pos, quat

def depth_to_xyz(depths, intrinsics):
    # Convert inputs to PyTorch tensors if they aren't already
    depths = torch.tensor(depths, dtype=torch.float32)
    intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
    
    # Batch dimensions
    batch_size, height, width = depths.shape
    
    # Create meshgrid of (u, v) coordinates for each depth map in the batch
    u, v = torch.meshgrid(torch.arange(height, dtype=torch.float32),
                          torch.arange(width, dtype=torch.float32))

    u = u.to(depths.device)
    v = v.to(depths.device)
    
    # Expand dimensions of u and v to match the batch size
    u = u.unsqueeze(0).expand(batch_size, -1, -1)
    v = v.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Extract intrinsic parameters for each batch
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    
    # Compute Z (depth) and ensure it has the batch dimension
    Z = depths
    
    # Compute X and Y for each depth map
    X = (u - cx.unsqueeze(1).unsqueeze(1)) * (Z / fx.unsqueeze(1).unsqueeze(1))
    Y = (v - cy.unsqueeze(1).unsqueeze(1)) * (Z / fy.unsqueeze(1).unsqueeze(1))
    
    # Stack to form the (X, Y, Z) coordinate tensor
    xyz = torch.stack((X, Y, Z), dim=-1)
    
    return xyz
