from datetime import datetime
from omni.isaac.lab.utils import configclass
from typing import Tuple, Callable, List, Optional
from dataclasses import field

@configclass
class GeneralCfg:
  # usd_path: str = "/home/arhan/Downloads/scene.usdz"
    usd_info: str = "./data/weirdlab_xform_mapping.yml"
    usd_path: str | None = None
    log_dir: str = "./logs"


@configclass
class SB3Cfg:
  seed: int = 42

  n_timesteps: float = 1e10
  policy: str = 'MlpPolicy'
  batch_size: int = 128
  n_steps: int =  512
  gamma: float = 0.99
  gae_lambda: float = 0.9
  n_epochs: int = 20
  ent_coef: float = 0.0
  sde_sample_freq: int = 4
  max_grad_norm: float = 0.5
  vf_coef: float = 0.5
  learning_rate: float = 3e-5
  use_sde: bool =  True
  clip_range: float = 0.4
  device: str = "cuda:0"
  policy_kwargs: str = '''dict(
                            log_std_init=-1,
                            ortho_init=False,
                            activation_fn=nn.ReLU,
                            net_arch=dict(pi=[256, 256], vf=[256, 256])
                        )'''

@configclass
class VideoCfg:
    enabled: bool = False
    viewer_resolution: Tuple[int, int] = (320, 240)
    viewer_eye: Tuple[float, float, float] = (-4.2, -2.5, 2.0)
    viewer_lookat: Tuple[float, float, float] = (3.8, 2.7, -1.2)
    video_folder: str = "videos"
    save_steps: int = 2500
    video_length: int = 250

###############################################################################
# GRASP CONFIG START
###############################################################################

@configclass
class DataConfig:
    root_dir: str = ''
    num_points: int = 25000
    num_object_points: int = 1024
    world_coord: bool = True
    num_rotations: int = 8
    grid_resolution: float = 0.01
    jitter_scale: float = 0.0
    contact_radius: float = 0.005
    robot_prob: float = 1.0
    offset_bins: List[float] = field(default_factory=lambda: [
        0, 0.00794435329, 0.0158887021, 0.0238330509,
        0.0317773996, 0.0397217484, 0.0476660972,
        0.055610446, 0.0635547948, 0.0714991435, 0.08
    ])
    synthetic_pcd: bool = True

@configclass
class SceneEncoderConfig:
    type: str = 'pointnet2_msg'
    num_points: int = 16384
    downsample: int = 4
    radius: float = 0.05
    radius_mult: int = 2
    use_rgb: bool = False

@configclass
class ObjectEncoderConfig:
    type: str = 'pointnet2_msg_cls'
    num_points: int = 1024
    downsample: int = 4
    radius: float = 0.05
    radius_mult: int = 2
    use_rgb: bool = False

@configclass
class ContactDecoderConfig:
    mask_feature: str = 'res0'
    in_features: List[str] = field(default_factory=lambda: ['res1', 'res2', 'res3'])
    place_feature: str = 'res4'
    object_in_features: List[str] = field(default_factory=lambda: ['res1', 'res2', 'res3'])
    embed_dim: int = 256
    feedforward_dim: int = 512
    num_scales: int = 3
    num_layers: int = 9
    num_heads: int = 8
    num_grasp_queries: int = 100
    num_place_queries: int = 8
    language_context_length: int = 0
    language_token_dim: int = 256
    use_attn_mask: bool = True
    use_task_embed: bool = True
    activation: str = 'GELU'

@configclass
class ActionDecoderConfig:
    use_embed: bool = False
    max_num_pred: Optional[int] = None
    hidden_dim: int = 256
    num_layers: int = 2
    num_params: int = 0
    activation: str = 'GELU'
    offset_bins: List[float] = field(default_factory=lambda: [
        0, 0.00794435329, 0.0158887021, 0.0238330509,
        0.0317773996, 0.0397217484, 0.0476660972,
        0.055610446, 0.0635547948, 0.0714991435, 0.08
    ])

@configclass
class MatcherConfig:
    object_weight: float = 2.0
    bce_weight: float = 5.0
    dice_weight: float = 5.0

@configclass
class GraspLossConfig:
    object_weight: float = 2.0
    not_object_weight: float = 0.1
    pseudo_ce_weight: float = 0.0
    bce_topk: int = 512
    bce_weight: float = 5.0
    dice_weight: float = 5.0
    deep_supervision: bool = True
    recompute_indices: bool = True
    adds_pred2gt: float = 100.0
    adds_gt2pred: float = 0.0
    adds_per_obj: bool = False
    contact_dir: float = 0.0
    approach_dir: float = 0.0
    offset: float = 1.0
    param: float = 1.0
    offset_bin_weights: List[float] = field(default_factory=lambda: [
        0.16652107, 0.21488856, 0.37031708, 0.55618503, 0.75124664,
        0.93943357, 1.07824539, 1.19423112, 1.55731375, 3.17161779
    ])

@configclass
class PlaceLossConfig:
    bce_topk: int = 1024
    bce_weight: float = 5.0
    dice_weight: float = 5.0
    deep_supervision: bool = True

@configclass
class OptimizerConfig:
    type: str = 'ADAMW'
    base_batch_size: int = 16
    base_lr: float = 0.0001
    backbone_multiplier: float = 1.0
    grad_clip: float = 0.01
    weight_decay: float = 0.05

@configclass
class TrainConfig:
    mask_thresh: float = 0.5
    num_gpus: int = 8
    port: str = '1234'
    batch_size: int = 16
    num_workers: int = 8
    num_epochs: int = 160
    print_freq: int = 25
    plot_freq: int = 50
    save_freq: int = 10
    checkpoint: Optional[str] = None
    log_dir: str = ''

@configclass
class EvalConfig:
    data_dir: str = ''
    checkpoint: str = '/home/arhan/Downloads/m2t2.pth'
    mask_thresh: float = 0.4
    object_thresh: float = 0.4
    num_runs: int = 1
    world_coord: bool = True
    surface_range: float = 0.02
    placement_height: float = 0.02
    placement_vis_radius: float = 0.3

@configclass
class M2T2Config:
    scene_encoder: SceneEncoderConfig = SceneEncoderConfig()
    object_encoder: ObjectEncoderConfig = ObjectEncoderConfig()
    contact_decoder: ContactDecoderConfig = ContactDecoderConfig()
    action_decoder: ActionDecoderConfig = ActionDecoderConfig()
    matcher: MatcherConfig = MatcherConfig()
    grasp_loss: GraspLossConfig = GraspLossConfig()
    place_loss: PlaceLossConfig = PlaceLossConfig()

@configclass
class GraspConfig:
    data: DataConfig = DataConfig()
    m2t2: M2T2Config = M2T2Config()
    optimizer: OptimizerConfig = OptimizerConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()

###############################################################################
# GRASP CONFIG END
###############################################################################

@configclass
class Config:
    general: GeneralCfg = GeneralCfg()
    sb3: SB3Cfg = SB3Cfg()
    video: VideoCfg = VideoCfg()
    grasp: GraspConfig = GraspConfig()
