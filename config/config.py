from dataclasses import dataclass, field
from typing import List
from .grasp import GraspConfig

@dataclass
class Config:
    usd_info_path: str = "./data/bowl2sink_xform_mapping.yml"
    usd_info: dict = field(default_factory=dict)  # Default to an empty dict
    usd_path: str = "./data/g60.usd"

    log_dir: str = "./logs"
    grasp: GraspConfig = GraspConfig()

    @dataclass
    class VideoConfig:
        enabled: bool = False
        viewer_resolution: List[int] = field(default_factory=lambda: 
                                             [320, 240])
        viewer_eye: List[float] = field(default_factory=lambda: 
                                        [-4.2, -2.5, 2.0])
        viewer_lookat: List[float] = field(default_factory=lambda: 
                                           [3.8, 2.7, -1.2])
        video_folder: str = 'videos'
        save_steps: int = 2500
        video_length: int = 250
    video: VideoConfig = VideoConfig()

    @dataclass
    class DataCollectionConfig:
        max_episodes: int = 200
    data_collection: DataCollectionConfig = DataCollectionConfig()

    @dataclass
    class ActionConfig:
        type: str = 'relative'
    actions: ActionConfig = ActionConfig()

