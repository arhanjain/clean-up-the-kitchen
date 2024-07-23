import torch
import omni.isaac.lab.utils.math as math

def pos_and_quat_to_matrix(pos, quat):
    '''
    Returns the 4x4 transformation matrix derived from the
    position and quaternion
    '''

    num_envs = pos.shape[0]

    rot_mat = math.matrix_from_quat(quat)
    pos = pos.unsqueeze(2)

    transformation = torch.cat((rot_mat, pos), dim=2).cpu()

    bottom_row = torch.tensor([0, 0, 0, 1]).expand(num_envs, 1, 4)
    transformation = torch.cat((transformation, bottom_row), dim=1).numpy()

    return transformation


