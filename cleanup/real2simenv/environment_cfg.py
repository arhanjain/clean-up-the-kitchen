import torch
import numpy as np
import os
import yaml
from cleanup.config import Config
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.utils import math as math_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from . import mdp
from omni.isaac.lab.sim.schemas import ArticulationRootPropertiesCfg
from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
)

from pxr import Usd, Sdf
from .utils import usd_utils, misc_utils
from .sensor import SiteCfg
import omni.isaac.lab.utils.math as math
from omni.isaac.core.utils.stage import get_current_stage

@configclass
class Real2SimSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """
    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
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
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0),
    )

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.1,
        # height=180,
        # width=320,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=25.0, focus_distance=400.0, horizontal_aperture=20.955,# clipping_range=(0.05, 2.0)
        ),
        # offset=CameraCfg.OffsetCfg(pos=(-0.23, 0.93, 0.66486), rot=(-0.2765, -0.20009, 0.5544, 0.75905), convention="opengl"), # old env
        offset=CameraCfg.OffsetCfg(pos=(-0.2411, -1.08517, 0.81276), rot=(0.81133, 0.50206, -0.15885, -0.25388), convention="opengl"), # kitchen
        # offset=CameraCfg.OffsetCfg(pos=(-0.59371, -1.10056, 0.76485), rot=(0.75146, 0.53087, -0.22605, -0.31999), convention="opengl"),
        semantic_filter="class:*",
        colorize_semantic_segmentation=False,
    )
    
    def __post_init__(self):
        super().__post_init__()
        self.load_kitchen_config()
        self.initialize_ee_frame()
    
    def load_kitchen_config(self):
        # Load the YAML file
        with open('/home/raymond/projects/clean-up-the-kitchen/cleanup/config/kitchen02.yaml', 'r') as f:
            kitchen_cfg = yaml.safe_load(f)
        current_path = os.getcwd()
        articulation_objects = kitchen_cfg['params'].get('ArticulationObject', {})
        rigid_objects = kitchen_cfg['params'].get('RigidObject', {})
        # Process the objects
        self.process_articulation_objects(articulation_objects, current_path)
        self.process_rigid_objects(rigid_objects, current_path)

    def process_articulation_objects(self, articulation_objects, current_path):
        for object_name, object_config in articulation_objects.items():
            axes = object_config["rot"]["axis"]
            angles = object_config["rot"]["angles"]
            quat = self.obtain_target_cam_quat(axes, angles)
            articulate_cfg = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/" + f"{object_config['name']}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=os.path.join(current_path, object_config['path']),
                    activate_contact_sensors=False,
                    rigid_props=RigidBodyPropertiesCfg(
                        disable_gravity=object_config.get("disable_gravity", False),
                        max_depenetration_velocity=5.0,
                    ),
                    articulation_props=ArticulationRootPropertiesCfg(
                        enabled_self_collisions=object_config.get("enabled_self_collisions", False),
                        solver_position_iteration_count=8,
                        solver_velocity_iteration_count=0
                    ),
                    scale=object_config.get("scale", [1, 1, 1])
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=object_config.get("pos", [0, 0, 0]),
                    rot=quat,
                    joint_pos=object_config.get("joints", {})
                ),
                actuators={
                    f"{object_config['name']}": ImplicitActuatorCfg(
                        joint_names_expr=[*object_config.get("joints", {}).keys()],
                        effort_limit=87.0,
                        velocity_limit=100.0,
                        stiffness=0.0,
                        damping=0.0,
                        friction=10.0,
                    ),
                },
            )
            setattr(self, object_config['name'], articulate_cfg)

    def process_rigid_objects(self, rigid_objects, current_path):
        for object_name, object_config in rigid_objects.items():
            axes = object_config["rot"]["axis"]
            angles = object_config["rot"]["angles"]
            quat = self.obtain_target_cam_quat(axes, angles)
            rigid_object_cfg = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"{object_name}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=os.path.join(current_path, object_config['path']),
                    scale=object_config.get('scale', [1, 1, 1])
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=object_config.get('pos', [0, 0, 0]),
                    rot=quat,
                )
            )
            setattr(self, object_name, rigid_object_cfg)
    
    def initialize_ee_frame(self):
        # Initialize the end-effector frame or any other components
        # marker_cfg = FRAME_MARKER_CFG.copy()
        # marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        # marker_cfg.prim_path = "/Visuals/FrameTransformer"
        # self.ee_frame = FrameTransformerCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        #     debug_vis=False,
        #     visualizer_cfg=marker_cfg,
        #     target_frames=[
        #         FrameTransformerCfg.FrameCfg(
        #             prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
        #             name="end_effector",
        #             offset=OffsetCfg(
        #                 pos=[0.0, 0.0, 0.0],
        #             ),
        #         ),
        #     ],
        # )
        self.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(
                prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="ee_tcp",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.1034), ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.046), ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.046), ),
                ),
            ],
        )

    def obtain_target_cam_quat(self, axis, angles):
        quat_list = []
        for index, cam_axis in enumerate(axis):
            euler_xyz = torch.zeros(3)
            
            euler_xyz[cam_axis] = angles[index]
            quat_list.append(
                math_utils.quat_from_euler_xyz(euler_xyz[0], euler_xyz[1],
                                            euler_xyz[2]))
        if len(quat_list) == 1:
            return torch.as_tensor(quat_list[0], dtype=torch.float16)
        else:
            target_quat = quat_list[0]
            for index in range(len(quat_list) - 1):

                target_quat = math_utils.quat_mul(quat_list[index + 1],
                                                target_quat)
            return torch.as_tensor(target_quat, dtype=torch.float16)

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.3, 0.4), roll=(0, 0), pitch=(np.pi, np.pi), yaw=(np.pi, np.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    body_joint_pos: DifferentialInverseKinematicsActionCfg | None = None

    finger_joint_pos: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )

    def setup(self, cfg: Config):
        relative_mode = cfg.actions.type == "relative"
        
        action_cfg = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=relative_mode, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        )

        setattr(self, "body_joint_pos", action_cfg)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        ee_pose = ObsTerm(
            func=mdp.ee_pose,
        )
        gripper_state = ObsTerm(
                func=mdp.gripper_state,
        )
        joint_state = ObsTerm(
            func = mdp.get_joint_info,
        )
        handle_pose = ObsTerm(
            func=mdp.get_handle_pose
        )

        rgb = ObsTerm(func=mdp.get_camera_data, params={"type": "rgb"})
        depth = ObsTerm(func=mdp.get_camera_data, params={"type": "distance_to_image_plane"})
        pcd = ObsTerm(func=mdp.get_point_cloud)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    
    # def setup(self, cfg):
    #     # TODO: dynamically add all objects from USD as state observations
    #     stage = Usd.Stage.Open(cfg.usd_path)
    #     for prim in stage.GetDefaultPrim().GetChildren():
    #         name = prim.GetName()
    #         if not name in ['light', 'scene_mat', 'sektion', ]: # temp fix
    #             term = ObsTerm(
    #                     func= mdp.object_pose_in_robot_root_frame,
    #                     params={"object_cfg": SceneEntityCfg(name)}
    #                     )
    #             setattr(self.policy, name, term)
        


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    randomize_ee_start_position = EventTerm(func=mdp.randomize_ee_start_position, mode="reset")
    
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "yaw": (-0.52, 0.52)
            },
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    #scurrent ee pos tensor([[0.3439, 0.0576, 0.5556]], device='cuda:0')            "pose_range": {"x": (-0.2, 0.2), "y": (-0.5, 0.5), "z": (-0.4, 0.4)},
    #             "velocity_range": {},
    #             "asset_cfg": SceneEntityCfg("camera"),
    #         },
    #         )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # ee_to_obj = RewTerm(
    #     func=mdp.object_ee_distance,
    #     params={
    #         "std": 0.1,
    #         "object_cfg": SceneEntityCfg("Xform_266")
    #         },
    #     weight=1.0
    # )
    #
    # # in the future turn this into a distance to command goal, not object
    # obj_to_site = RewTerm(
    #     func=mdp.object_to_object_distance,
    #     params={
    #         "std":0.3,
    #         "object1_cfg": SceneEntityCfg("SiteXform_267"),
    #         "object2_cfg": SceneEntityCfg("Xform_266"),
    #     },
    #     weight=2.0
    # )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # object_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum, params={"minimum_height": -0.3, "asset_cfg": SceneEntityCfg("Xform_266")}
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    # )




@configclass
class Real2SimCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # # Scene settings
    scene = Real2SimSceneCfg(num_envs=4, env_spacing=3)
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
        self.decimation = 10 # 10 hz for control/step
        self.episode_length_s = 20
        # simulation settings
        self.sim.dt = 0.01  # 100Hz for physx

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

    def setup(self, cfg):
        for _, attr in self.__dict__.items():
            if hasattr(attr, 'setup') and callable(getattr(attr, 'setup')):
                attr.setup(cfg)