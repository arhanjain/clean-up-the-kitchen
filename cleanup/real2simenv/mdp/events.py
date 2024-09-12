import torch 
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers import SceneEntityCfg
import omni.isaac.lab.utils.math as math_utils


def reset_cam(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # get default root state
    # root_states = asset.data.default_root_state[env_ids].clone()
    pos = torch.tensor([[-0.355, 0.855, 0.788]])
    rot = torch.tensor([[0.402, 0.245, -0.459, -0.753]])
    root_states = torch.cat([pos, rot, torch.zeros(1, 7)], dim=-1).cuda()

    # velocities
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    # poses
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]

    # solve for orientation as a (lookat)
    look_at = torch.tensor([[0.5, 0.0, 0.0]]).cuda()
    direction_vector = -(look_at - positions)
    direction_vector = direction_vector / torch.norm(direction_vector, dim=-1, keepdim=True)
    up_vector = torch.tensor([[0.0, 0.0, 1.0]]).cuda()
    right_vector = torch.cross(up_vector, direction_vector)
    right_vector = right_vector / torch.norm(right_vector, dim=-1, keepdim=True)
    up_vector = torch.cross(direction_vector, right_vector)
    up_vector = up_vector / torch.norm(up_vector, dim=-1, keepdim=True)
    orientations = math_utils.quat_from_matrix(torch.stack([right_vector, up_vector, direction_vector], dim=-1))

    # set into the physics simulation
    asset.set_world_poses(positions, orientations, env_ids, convention="opengl")
