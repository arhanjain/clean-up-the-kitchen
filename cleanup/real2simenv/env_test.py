import numpy as np
from config.config import Config
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
from . import mdp

from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
import omni.isaac.lab.utils.math as math

from pxr import Usd, Sdf
import torch
from .utils import usd_utils, misc_utils
from .sensor import SiteCfg

@configclass
class TestCfg(InteractiveSceneCfg):
    """ConfiguratiounitsResolven for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """
    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
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
        height=180,
        width=320,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=25.0, focus_distance=400.0, horizontal_aperture=20.955, # clipping_range=(0.1, 2.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.10, -0.7, 0.32), rot=(0.77, 0.5, -0.23, -0.33), convention="opengl"),
        semantic_filter="class:*",
        colorize_semantic_segmentation=False,
    )

    # scene = AssetBaseCfg(
    #         prim_path="{ENV_REGEX_NS}/scene",
    #         init_state=RigidObjectCfg.InitialStateCfg(pos=[0, 0, 0]),
    #         spawn=UsdFileCfg(
    #             usd_path="./data/g60.usd"
    #     ))

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
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
             ),
            ],
        )



    def setup(self, cfg: Config):

    
        usd_info = cfg.usd_info

        # parse and add USD
        objs = {}
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"



        stage = Usd.Stage.Open(cfg.usd_path)
        for prim in stage.GetDefaultPrim().GetChildren():
            sub_path = prim.GetPath().pathString
            pos = prim.GetAttribute("xformOp:translate").Get()
            rot = prim.GetAttribute("xformOp:orient").Get()
            pos = (pos[0], pos[1], pos[2])
            quat = torch.tensor((rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2]))

            euler = math.euler_xyz_from_quat(quat.unsqueeze(0))
            quat = math.quat_from_euler_xyz(euler[0]+np.pi/2, euler[1], euler[2])
            quat = quat[0]

            name = prim.GetName()
            objs[name] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=quat),
                spawn=usd_utils.CustomRigidUSDCfg(
                    usd_path=cfg.usd_path,
                    usd_sub_path=sub_path,
                    semantic_tags=[("class", name)],
                )
            )

        for k, v in objs.items():
            setattr(self, k, v)



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

    body_joint_pos: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            )

    finger_joint_pos: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )



    def setup(self, cfg):
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

        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_pose = ObsTerm(
            func=mdp.ee_pose,
        )
        gripper_state = ObsTerm(
                func=mdp.gripper_state,
                        )

        rgb = ObsTerm(func=mdp.get_camera_data, params={"type": "rgb"})
        # depth = ObsTerm(func=mdp.get_camera_data, params={"type": "distance_to_image_plane"})
        pcd = ObsTerm(func=mdp.get_point_cloud)
        # shoulder_cam = ObsTerm(
        #         func=mdp.camera_rgb,
        #         )
        # object_position = ObsTerm(
        #     func=mdp.object_position_in_robot_root_frame,
        #     params={"object_cfg": SceneEntityCfg("blue_cube")}
        

        # object_position = ObsTerm(
        #     func=mdp.object_position_in_robot_root_frame,
        #     params={"object_cfg": SceneEntityCfg("Xform_266")}
        # )
        # target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})

        # actions = ObsTerm(func=mdp.last_action)
        # rgb, seg, depth = ObsTerm(func=mdp.get_camera_data)
        # seg = ObsTerm(func=mdp.get_camera_data, params={"type": "semantic_segmentation"})
        # depth = ObsTerm(func=mdp.get_camera_data, params={"type": "distance_to_image_plane"})


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    
    def setup(self, cfg):
        usd_info = cfg.usd_info
        # for name in usd_info["xforms"]:
        #     setattr(self.policy, name, ObsTerm(
        #             func=mdp.object_position_in_robot_root_frame, 
        #             params={"object_cfg": SceneEntityCfg(name)}
        #             ))
        # for name in usd_info["sites"]:
        #     setattr(self.policy, name, ObsTerm(
        #             func=mdp.object_position_in_robot_root_frame, 
        #             params={"object_cfg": SceneEntityCfg(name)}
        #             ))


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    #
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0),
                           "pitch": (-np.pi, np.pi)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("newcube"),
        },
    )


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
class TestEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # # Scene settings
    scene = TestCfg(num_envs=4, env_spacing=3)
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
        self.episode_length_s = 8.0
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
