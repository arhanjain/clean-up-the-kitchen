import open3d as o3d
import numpy as np
import random

from moviepy.editor import ImageSequenceClip
from pathlib import Path
import argparse 
from tqdm import tqdm

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--data_path", type=str, default="./data/nvidia_poop/")
args = arg_parser.parse_args()


# data = np.load("./data/nvidia_trial/episode_0.npz", allow_pickle=True)

# viz = o3d.visualization.Visualizer()
# viz.create_window()
#
# geometry = o3d.geometry.PointCloud()
# geometry.points = o3d.utility.Vector3dVector(data["observations"][0]["policy"]["pcd"].reshape(-1, 3))
# viz.add_geometry(geometry)
folder = Path(args.data_path)
(folder/"viz").mkdir(exist_ok=True)
trajs = list(folder.glob("*.npz"))
chosen = random.sample(trajs, min(10, len(trajs)))
# chosen = ["./data/nvidia_trial/episode_22.npz"]
for idx,path in enumerate(tqdm(chosen)):
    data = np.load(path, allow_pickle=True)
    vid = []
    for i in range(1, data["observations"].shape[0]):
        img = data["observations"][i]["policy"]["rgb"].squeeze()[:, :, :3]
        vid.append(img)
    ImageSequenceClip(vid, fps=30).write_videofile(str(folder/"viz"/f"{idx}.mp4"), codec="libx264", fps=10)
    # geometry.points = o3d.utility.Vector3dVector(data["observations"][i]["policy"]["pcd"].reshape(-1, 3))
    # viz.update_geometry(geometry)
    # viz.poll_events()
    # viz.update_renderer()

# # vid = np.array(vid)

# ImageSequenceClip(vid, fps=30).write_videofile("data/viz.mp4", codec="libx264", fps=10)
 
