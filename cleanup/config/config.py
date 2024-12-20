from dataclasses import dataclass, field
from typing import List, Tuple
from .grasp import GraspConfig

@dataclass
class Config:
    usd_path: str = "./data/g60.usd"
    log_dir: str = "./logs"
    grasp: GraspConfig = GraspConfig()

    @dataclass
    class DeployConfig:
        replay: bool = False
        dataset: str = ""
        checkpoint: str = ""
    deploy: DeployConfig = DeployConfig()

    @dataclass
    class VideoConfig:
        enabled: bool = False
        # viewer_resolution: List[int] = field(default_factory=lambda: 
        #                                      [320, 240])
        viewer_resolution: Tuple[int, int] = (320, 240)
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
        ds_name: str = "test"
        max_episodes: int = 10
        save: bool = True
    data_collection: DataCollectionConfig = DataCollectionConfig()

    @dataclass
    class ActionConfig:
        type: str = 'relative' # relative, absolute
    actions: ActionConfig = ActionConfig()

