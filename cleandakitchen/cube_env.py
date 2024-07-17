
import pickle
import torch
import numpy as np
from PIL import Image
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
# from omni.isaac.lab.envs.mdp import UniformPoseCommandCfg
from . import mdp

# from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG  # isort: skip
from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip

from gymnasium import Wrapper

@configclass
class CubeSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # robot.init_state.pos = (0.0, 0.0, 4.05)
    # robot = None

    # target object: will be populated by agent env cfg
    # object: RigidObjectCfg = MISSING
    # Set Cube as object
    # person = UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/People/Characters/original_male_adult_police_04/male_adult_police_04.usd",
    #         scale=(0.8, 0.8, 0.8),
    #         rigid_props=RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=1,
    #             max_angular_velocity=1000.0,
    #             max_linear_velocity=1000.0,
    #             max_depenetration_velocity=5.0,
    #             disable_gravity=False,
    #         ),
    #         semantic_tags=[("class", "cube")]
    # )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            semantic_tags=[("class", "cube")]
        ),
    )

    # test = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Test",
    #     spawn=UsdFileCfg(
    #         usd_path="/home/arhan/Downloads/USDAssets/obj2sink_1.usd",
    #     )
    # )

    # kitchen = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Kitchen",
    #     spawn=UsdFileCfg(
    #         usd_path="/home/arhan/Downloads/kitchen.usdz",
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         # scale=(0.1,0.1,0.1)
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0, 0, 0], rot=[0.5,0.5, -0.5,-0.5])
    # )



    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            # semantic_tags=[("class", "table")]
            ),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=25.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.0)
        ),
        # offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        offset=CameraCfg.OffsetCfg(pos=(2.0, 0.0, 1.0), rot=(-0.612, -0.353, -0.353, -0.612), convention="opengl"),
        semantic_filter="class:*",
        colorize_semantic_segmentation=False,
    )

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # # end-effector sensor: will be populated by agent env cfg
        # ee_frame: FrameTransformerCfg = MISSING
        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0, 0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    body_joint_pos: mdp.JointPositionActionCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
    )
    # body_joint_pos = mdp.JointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=["panda_joint.*"],
    #     scale=1.0,
    #     use_default_offset=True,
    # )
    finger_joint_pos: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )




@configclass
class CubeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: CubeSceneCfg = CubeSceneCfg(num_envs=2, env_spacing=3)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


class TestWrapper(Wrapper):
    def __init__(self, env: ManagerBasedRLEnv):
        if not isinstance(env.unwrapped, ManagerBasedRLEnv):
            raise ValueError("Environment must be a ManagerBasedRLEnv...")

        # self.env = env 
        # self.scene = env.scene
        super().__init__(env)
    
    def get_joint_info(self):
        # Specify robot-specific parameters
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        robot_entity_cfg.resolve(self.scene)

        joint_pos = self.scene["robot"].data.joint_pos[:, robot_entity_cfg.joint_ids]
        joint_vel = self.scene["robot"].data.joint_vel[:, robot_entity_cfg.joint_ids]
        joint_names = self.scene["robot"].data.joint_names

        return joint_pos, joint_vel, joint_names

    def get_camera_data(self):
        rgb = self.scene["camera"].data.output["rgb"]
        seg = self.scene["camera"].data.output["semantic_segmentation"]
        depth = self.scene["camera"].data.output["distance_to_image_plane"]
        depth = np.clip(depth.cpu().numpy(), None, 1e5)

        save_dir = "/home/arhan/projects/IsaacLab/source/standalone/clean-up-the-kitchen/data/"
        # save rgb
        Image.fromarray(rgb[0].cpu().numpy()).convert("RGB").save(f"{save_dir}/rgb.png")
        # save depth
        np.save(f"{save_dir}/depth.npy", depth[0])
        # save seg
        mask = torch.clamp(seg-1, max = 1).cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(mask[0], mode="L").save(f"{save_dir}/seg.png")

        # metadata
        metadata = {}
        intrinsics = self.scene["camera"].data.intrinsic_matrices[0]

        # camera pose
        cam_pos = self.scene["camera"].data.pos_w
        cam_quat = self.scene["camera"].data.quat_w_ros

        robot_pos = self.scene["robot"].data.root_state_w[:, :3]
        robot_quat = self.scene["robot"].data.root_state_w[:, 3:7]

        cam_pos_r, cam_quat_r = math.subtract_frame_transforms(
            robot_pos, robot_quat,
            cam_pos, cam_quat
        )
        cam_rot_mat_r = math.matrix_from_quat(cam_quat_r)
        cam_pos_r = cam_pos_r.unsqueeze(2)

        transformation = torch.cat((cam_rot_mat_r, cam_pos_r), dim=2).cpu()

        bottom_row = torch.tensor([0,0,0,1]).expand(self.num_envs, 1, 4)
        transformation = torch.cat((transformation, bottom_row), dim=1).numpy()


        # filler from existing file
        ee_pose = np.array([[ 0.02123945,  0.82657526,  0.56242531,  0.18838109],
        [ 0.99974109, -0.02215279, -0.00519713, -0.01743025],
        [ 0.00816347,  0.56239007, -0.82683176,  0.6148137 ],
        [ 0.        ,  0.        ,  0.        ,  1.        ]])
        scene_bounds = np.array([-0.4, -0.8, -0.2, 1.2, 0.8, 0.6])

        metadata["intrinsics"] = intrinsics.cpu().numpy()
        metadata["camera_pose"] = transformation[0]
        metadata["ee_pose"] = ee_pose
        metadata["label_map"] = None
        # metadata["scene_bounds"] = scene_bounds

        with open(f"{save_dir}/meta_data.pkl", "wb") as f:
            pickle.dump(metadata, f)

        return rgb, seg, depth


    def goal_pose(self):
        return mdp.generated_commands(self.env, "object_pose")
    # def object_pos(self):
    #     return self.env.