import open3d as o3d
import numpy as np

from moviepy.editor import ImageSequenceClip


data = np.load("./data/g60_pick_remastered/episode_3.npz", allow_pickle=True)

# viz = o3d.visualization.Visualizer()
# viz.create_window()
#
# geometry = o3d.geometry.PointCloud()
# geometry.points = o3d.utility.Vector3dVector(data["observations"][0]["policy"]["pcd"].reshape(-1, 3))
# viz.add_geometry(geometry)

vid = []
for i in range(1, data["observations"].shape[0]):
    breakpoint()
    img = data["observations"][i]["policy"]["rgb"].squeeze()[:, :, :3]
    vid.append(img)
    # geometry.points = o3d.utility.Vector3dVector(data["observations"][i]["policy"]["pcd"].reshape(-1, 3))
    # viz.update_geometry(geometry)
    # viz.poll_events()
    # viz.update_renderer()

# vid = np.array(vid)

ImageSequenceClip(vid, fps=30).write_videofile("data/g60_pick_remastered.mp4", codec="libx264", fps=10)
 
