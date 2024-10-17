import torch 
from omni.isaac.lab.envs import ManagerBasedRLEnv, ManagerBasedEnv
from omni.isaac.lab.managers import SceneEntityCfg
import omni.isaac.lab.utils.math as math_utils
import numpy as np
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.sensors import FrameTransformer

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

def randomize_ee_start_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    for _ in range(15):
        env.sim.step()
    robot: Articulation = env.unwrapped.scene[robot_cfg.name]
    num_envs = env.unwrapped.scene.num_envs
    device = env.unwrapped.device
    ee_idx = robot.data.body_names.index("panda_hand")
    # Set current ee pose to what it should start as under the ideal
    # joint position so that IK works correctly

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w

    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot_pos_w, robot_quat_w,
        ee_pos_w, ee_quat_w
    )
    print("Current ee_pos", ee_pos_b)

    random_ee_positions = np.zeros((num_envs, 3))
    random_x = torch.FloatTensor(num_envs,).uniform_(0.25, 0.45)
    random_y = torch.FloatTensor(num_envs,).uniform_(-0.25, 0.35)
    random_z = torch.FloatTensor(num_envs,).uniform_(0.15, 0.3)

    # random_x = np.random.uniform(low=0.25, high=0.45, size=(num_envs,))
    # random_y = np.random.uniform(low=-0.25, high=0.35, size=(num_envs,))
    # random_z = np.random.uniform(low=-0.15, high=0.3, size=(num_envs,))

    print("random_x", random_x)
    print("random_y", random_y)
    print("random_z", random_z)


    random_ee_positions[:, 0]= random_x
    random_ee_positions[:, 1]= random_y
    random_ee_positions[:, 2]= random_z
    random_ee_positions = torch.tensor(random_ee_positions, device=device)

    ik_cfg = DifferentialIKControllerCfg(
        command_type="position", use_relative_mode=False, ik_method="pinv")
    ik_controller = DifferentialIKController(
        cfg=ik_cfg, num_envs=num_envs, device=device
    )
    ik_controller.set_command(command=random_ee_positions, ee_quat=ee_quat_b)

    jacobi_body_idx = ee_idx - 1
    joint_ids = list(range(robot.num_joints))
    print("num joints", robot.num_joints)
    print("joints", robot.data.joint_names)
    jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, joint_ids]

    target_joint_positions = ik_controller.compute(ee_pos=ee_pos_b, ee_quat=ee_quat_b,
        jacobian=jacobian, joint_pos=robot.data.joint_pos)
    robot.set_joint_position_target(
        target=target_joint_positions, joint_ids=joint_ids)
    
    print("Moving robot to position", random_ee_positions)
    robot.write_data_to_sim()
    # Give the robot time to move to the start pos
    for _ in range(15):
        env.sim.step()
        env.scene.update(env.sim.get_physics_dt())

