import torch
import omni.isaac.lab.utils.math as math

def relative_action_to_target(env, desired_pose):
    '''
    Compute relative action to target pose.

    Parameters
    ----------
    env : Real2SimRLEnv
    desired_pose:
        Desired pose of the target in the scene. It is a torch.Tensor of shape (N, 7), 3 for position and 4 for quaternion.

    Returns
    -------
    rel_pos : torch.Tensor((N, 3)), dtype=torch.float32)
        Relative position of the target in the scene.
    rel_euler : torch.Tensor((N, 3)), dtype=torch.float32)
        Relative euler angles of the target in the scene
    '''
    des_pos, des_quat = desired_pose[:, :3], desired_pose[:, 3:]
    curr_pos = env.unwrapped.scene["ee_frame"].data.target_pos_source[:, 0]
    curr_quat = env.unwrapped.scene["ee_frame"].data.target_quat_source[:, 0]

    rel_pos, rel_euler = math.compute_pose_error(
            curr_pos, curr_quat,
            des_pos, des_quat,
            rot_error_type="axis_angle"
            )
    return rel_pos, rel_euler
