import torch 
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers import SceneEntityCfg
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg


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

def reset_ee_pose(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        desired_pose_offset_range: dict[str, tuple[float, float]],
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        body_name="wx250s_ee_gripper_link",
        ):
    robot = env.scene[robot_cfg.name]
    num_envs = env.scene.num_envs
    ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
            )
    ik_controller = DifferentialIKController(ik_cfg, num_envs=num_envs, device=env.device)
    quat = math_utils.quat_from_euler_xyz(torch.tensor([0]), torch.tensor([1.57]), torch.tensor([0]))

    # default_pose = robot.data.default_root_state[env_ids].clone()
    range_list = [desired_pose_offset_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=robot.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 6), device=robot.device)
    
    # positions = default_pose[:, 0:3]
    positions = rand_samples[:, :3]
    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    # positions = default_pose[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, :3]
    # orientation_deltas = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    # orientations = math_utils.quat_mul(default_pose[:, 3:7], orientation_deltas)

    des_pose = torch.cat([positions, orientations], dim=-1)
    print(f"Resetting ee pose to orientation: {rand_samples[:, 3:]}")
    # breakpoint()

    
    # des_pose = torch.cat([torch.tensor([[0.3, 0.0, 0.15]]), quat], dim=-1).repeat(num_envs, 1).to(env.device)
    # des_pose = desired_pose

    # set the commanded ee_pose
    ik_controller.ee_pos_des = des_pose[:, :3]
    ik_controller.ee_quat_des = des_pose[:, 3:]

    # compute desired joint pos
    body_ids, body_names = robot.find_bodies(body_name)
    body_idx, body_name = body_ids[0], body_names[0]
    ee_pose_w = robot.data.body_state_w[:, body_idx, :7]
    root_pose_w = robot.data.root_state_w[:, :7]

    ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )

    jacobi_body_idx = body_idx - 1 # base is fixed, jacobian for base is not computed
    joint_ids = list(range(robot.num_joints))
    jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, joint_ids]
    joint_pos = robot.data.joint_pos
    des_joint_pos = ik_controller.compute(
            ee_pos=ee_pose_b, ee_quat=ee_quat_b, jacobian=jacobian, joint_pos=joint_pos
            ) # curr eepos, eequat, jacobian, joint_pos
    des_joint_pos = des_joint_pos[env_ids].clone()
    des_joint_vel = robot.data.default_joint_vel[env_ids].clone()
    robot.write_joint_state_to_sim(des_joint_pos, des_joint_vel, env_ids=env_ids)

