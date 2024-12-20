
import torch
import numpy as np
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
from . import mdp

from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip

from pxr import Usd, Sdf
from .utils import usd_utils, misc_utils
from .sensor import SiteCfg
import omni.isaac.lab.utils.math as math

@configclass
class Real2SimSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
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
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0),
    )
    
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.1,
        # height=180,
        # width=320,
        height=224,
        width=224,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=25.0, focus_distance=400.0, horizontal_aperture=20.955,# clipping_range=(0.05, 2.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.71, 0.955, 1.005), rot=(-0.41, -0.25, 0.45, 0.748), convention="opengl"),
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
                        pos=[0.0, 0.0, 0.0],
                    ),
             ),
            ],
        )

    
    def setup(self, cfg: Config):
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
            #
            # euler = math.euler_xyz_from_quat(quat.unsqueeze(0))
            # quat = math.quat_from_euler_xyz(euler[0]+np.pi/2, euler[1], euler[2])
            # quat = quat[0]
            #
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

    # object_pose = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name="panda_hand",  # will be set by agent env cfg
    #     resampling_time_range=(5.0, 5.0),
    #     debug_vis=False,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.3, 0.4), roll=(0, 0), pitch=(np.pi, np.pi), yaw=(np.pi, np.pi)
    #     ),
    # )


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
        if cfg.actions.type == "joint_pos":
            action_cfg = mdp.JointPositionActionCfg(
                use_default_offset=False,
                asset_name="robot",
                joint_names=["panda_joint.*"],
            )
        else:
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

        rgb = ObsTerm(func=mdp.get_camera_data, params={"type": "rgb"})
        # depth = ObsTerm(func=mdp.get_camera_data, params={"type": "distance_to_image_plane"})
        # pcd = ObsTerm(func=mdp.get_point_cloud)


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    
    def setup(self, cfg):
        # TODO: dynamically add all objects from USD as state observations
        stage = Usd.Stage.Open(cfg.usd_path)
        for prim in stage.GetDefaultPrim().GetChildren():
            name = prim.GetName()
            term = ObsTerm(
                    func= mdp.object_pose_in_robot_root_frame,
                    params={"object_cfg": SceneEntityCfg(name)}
                    )
            setattr(self.policy, name, term)


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    randomize_ee = EventTerm(
            func=mdp.reset_ee_pose,
            mode="reset",
            params={
                "desired_pose_offset_range": {
                    "x": (0.3, 0.5),
                    "y": (-0.2, 0.2),
                    "z": (0.35, 0.55),
                    "roll": ((10/10)*np.pi, (10/10)*np.pi),
                    "pitch": (-(0/10)*np.pi, (0/10)*np.pi),
                    "yaw": (-(0/10)*np.pi, (0/10)*np.pi),
                    },
                "robot_cfg": SceneEntityCfg("robot"),
                "body_name": "panda_hand",
                }
            )
    randomize_cube = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0., 0.), "y": (-0.1, 0.1), "z": (0., 0.)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("cube"),
                },
            )

    # randomize_joints = EventTerm(
    #         func= mdp.reset_joints_by_offset,
    #         mode="reset",
    #         params={
    #             "position_range": (-0.5, 0.1),
    #             "velocity_range": (-0., 0.),
    #             }
    #         )
    # randomize_robot_root = EventTerm(

    # reset_carrot = EventTerm(
    #         func=mdp.reset_root_state_with_random_orientation,
    #         mode="reset",
    #         params={
    #             # "pose_range": {"x": (-0.1, 0.1), "y": (-0.12, 0.2), "z": (0.2, 0.22)}, # normal
    #             "pose_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (0.0, 0.0)}, # easy
    #             "velocity_range": {},
    #             "asset_cfg": SceneEntityCfg("carrot"),
    #         },
    #         )
    # reset_camera = EventTerm(
    #         func=mdp.reset_cam,
    #         mode="reset",
    #         params={
    #             "pose_range": {"x": (-0.2, 0.2), "y": (-0.5, 0.5), "z": (-0.4, 0.4)},
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

    success = DoneTerm(
            func=mdp.object_reached_pos,
            params={
                "threshold": 0.15,
                "object_cfg": SceneEntityCfg("cube"),
            }
            )
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
    scene = Real2SimSceneCfg(num_envs=4, env_spacing=10)
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
