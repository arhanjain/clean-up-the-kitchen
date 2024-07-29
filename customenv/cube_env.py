
from dataclasses import MISSING
import torch
import re
import numpy as np
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkers, VisualizationMarkersCfg
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math
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
            focal_length=25.0, focus_distance=400.0, horizontal_aperture=20.955, #clipping_range=(0.1, 2.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.7, 1.1, 1.0), rot=(0.4, 0.27, -0.45, -0.75), convention="opengl"),
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
            debug_vis=True,
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

    
    def setup(self, cfg):
        # parse and add USD
        objs = {}
        usd_path = cfg["usd_path"]
        usd_stage = Usd.Stage.Open(usd_path)
        
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        site_pattern = r"^SiteXform_\d+$"
        xform_263_pos = None
        for entry in usd_stage.GetDefaultPrim().GetChildren()[:-1]:
            name = entry.GetName()
            if re.match(site_pattern, name):
                # handle site
                transform = entry.GetChildren()[-1].GetAttribute("xformOp:transform").Get()
                transform = torch.tensor(transform)
                pos, _ = misc_utils.GUI_matrix_to_pos_and_quat(transform)

                offset = pos - xform_263_pos

                # number = name.split("_")[-1]
                objs[name] = SiteCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Xform_{263}",
                    debug_vis=True,
                    offset=offset
                )
            else:
                transform = entry.GetChildren()[-1].GetAttribute("xformOp:transform").Get()
                transform = torch.tensor(transform)
                pos, quat = misc_utils.GUI_matrix_to_pos_and_quat(transform)
                # pos, quat = (0,0,0), (1,0,0,0)
                objs[name] = RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{name}",
                    init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=quat),
                    spawn=usd_utils.CustomRigidUSDCfg(
                        usd_path=usd_path,
                        usd_sub_path=entry.GetPath().pathString,
                        rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True if name == "Xform_263" else False),
                        collision_props=CollisionPropertiesCfg(),
                        semantic_tags=[("class", "obj")],
                    )
                )

                if name == "Xform_263":
                    xform_263_pos = pos

        

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
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.15, 0.15), roll=(0, 0), pitch=(np.pi, np.pi), yaw=(np.pi, np.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    body_joint_pos: mdp.DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
    )

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

        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_position = ObsTerm(
            func=mdp.ee_pose,
        )
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("Xform_266")}
        )
        # target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})

        # actions = ObsTerm(func=mdp.last_action)
        # rgb, seg, depth = ObsTerm(func=mdp.get_camera_data)
        # rgb = ObsTerm(func=mdp.get_camera_data, params={"type": "rgb"})
        # seg = ObsTerm(func=mdp.get_camera_data, params={"type": "semantic_segmentation"})
        # depth = ObsTerm(func=mdp.get_camera_data, params={"type": "distance_to_image_plane"})


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object", body_names="Object"),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    ee_to_obj = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.1,
            "object_cfg": SceneEntityCfg("Xform_266")
            },
        weight=1.0
    )

    # in the future turn this into a distance to command goal, not object
    obj_to_site = RewTerm(
        func=mdp.object_to_object_distance,
        params={
            "std":0.3,
            "object1_cfg": SceneEntityCfg("SiteXform_267"),
            "object2_cfg": SceneEntityCfg("Xform_266"),
        },
        weight=2.0
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.3, "asset_cfg": SceneEntityCfg("Xform_266")}
    )


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
class CubeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # # Scene settings
    scene: CubeSceneCfg = CubeSceneCfg(num_envs=4, env_spacing=3)
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

    def setup(self, cfg):
        for _, attr in self.__dict__.items():
            if hasattr(attr, 'setup') and callable(getattr(attr, 'setup')):
                attr.setup(cfg)
