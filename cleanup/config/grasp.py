from dataclasses import dataclass
from typing import List, Optional
from dataclasses import dataclass, field

@dataclass 
class GraspConfig:
    
    synthetic_pcd: bool = False

    ### Everything below this line is default by M2T2 authors ###
    @dataclass
    class GraspData:
        root_dir: str = ''
        num_points: int = 16384
        num_object_points: int = 1024
        world_coord: bool = True
        num_rotations: int = 8
        grid_resolution: float = 0.01
        jitter_scale: float = 0.0
        contact_radius: float = 0.005
        robot_prob: float = 1.0
        offset_bins: List[float] = field(default_factory=lambda: 
                                         [0, 0.00794435329, 0.0158887021,
                                          0.0238330509, 0.0317773996, 0.0397217484,
                                          0.0476660972, 0.055610446, 0.0635547948,
                                          0.0714991435, 0.08])
    data: GraspData = GraspData()

    @dataclass
    class M2T2:

        @dataclass
        class SceneEncoder:
            type: str = 'pointnet2_msg'
            num_points: int = 16384
            downsample: int = 4
            radius: float = 0.05
            radius_mult: int = 2
            use_rgb: bool = False
        scene_encoder: SceneEncoder = SceneEncoder()

        @dataclass
        class ObjectEncoder:
            type: str = 'pointnet2_msg_cls'
            num_points: int = 1024
            downsample: int = 4
            radius: float = 0.05
            radius_mult: int = 2
            use_rgb: bool = False
        object_encoder: ObjectEncoder = ObjectEncoder()

        @dataclass
        class ContactDecoder:
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
        contact_decoder: ContactDecoder = ContactDecoder()

        @dataclass
        class ActionDecoder:
            use_embed: bool = False
            max_num_pred: Optional[int] = None
            hidden_dim: int = 256
            num_layers: int = 2
            num_params: int = 0
            activation: str = 'GELU'
            offset_bins: List[float] = field(default_factory=lambda: 
                                             [0, 0.00794435329, 0.0158887021, 
                                              0.0238330509, 0.0317773996, 0.0397217484, 
                                              0.0476660972, 0.055610446, 0.0635547948, 
                                              0.0714991435, 0.08])
        action_decoder: ActionDecoder = ActionDecoder()

        @dataclass
        class Matcher:
            object_weight: float = 2.0
            bce_weight: float = 5.0
            dice_weight: float = 5.0
        matcher: Matcher = Matcher()

        @dataclass
        class GraspLoss:
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
            offset_bin_weights: List[float] = field(default_factory=lambda: 
                                                    [0.16652107, 0.21488856, 0.37031708, 
                                                     0.55618503, 0.75124664, 0.93943357, 
                                                     1.07824539, 1.19423112, 1.55731375, 
                                                     3.17161779])
        grasp_loss: GraspLoss = GraspLoss()

        @dataclass
        class PlaceLoss:
            bce_topk: int = 1024
            bce_weight: float = 5.0
            dice_weight: float = 5.0
            deep_supervision: bool = True
        place_loss: PlaceLoss = PlaceLoss()
    m2t2: M2T2 = M2T2()




    @dataclass
    class Optimizer:
        type: str = 'ADAMW'
        base_batch_size: int = 16
        base_lr: float = 0.0001
        backbone_multiplier: float = 1.0
        grad_clip: float = 0.01
        weight_decay: float = 0.05
    optimizer: Optimizer = Optimizer()

    @dataclass
    class Train:
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
    train: Train = Train()

    @dataclass
    class Eval:
        data_dir: str = ''
        checkpoint: str = '/home/raymond/Downloads/m2t2.pth'
        mask_thresh: float = 0.4
        object_thresh: float = 0.4
        num_runs: int = 1
        world_coord: bool = True
        surface_range: float = 0.02
        placement_height: float = 0.02
        placement_vis_radius: float = 0.3
    eval: Eval = Eval()

