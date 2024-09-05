import torch
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab import markers
from omni.isaac.lab.managers.scene_entity_cfg import SceneEntityCfg
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.sensors import SensorBase
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import convert_quat
import omni.physics.tensors.impl.api as physx
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .site_cfg import SiteCfg

@configclass
class SiteData:
    root_pos_w: torch.Tensor | None = None
    root_quat_w: torch.Tensor | None = None


class Site(SensorBase):

    def __init__(self, cfg: 'SiteCfg'):
        super().__init__(cfg)
        self._data: SiteData = SiteData()
    
    @property
    def data(self) -> SiteData:
        self._update_outdated_buffers()
        return self._data
    
    def _initialize_impl(self):
        super()._initialize_impl()

        obj_name_regex = f"{self.cfg.prim_path}/*"
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")
        self._obj_physx_view = self._physics_sim_view.create_rigid_body_view(obj_name_regex.replace(".*", "*"))

        self._data.root_pos_w = torch.zeros(self._num_envs, 3, device=self._device)
        self._data.root_quat_w = torch.zeros(self._num_envs, 4, device=self._device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        
        transforms = self._obj_physx_view.get_transforms()
        # Convert quaternions as PhysX uses xyzw form
        transforms[:, 3:] = convert_quat(transforms[:, 3:], to="wxyz")

        offset = torch.tensor(self.cfg.offset, device=self._device).repeat(self._num_envs, 1)

        self._data.root_pos_w = transforms[:, :3] + offset
        self._data.root_quat_w = transforms[:, 3:]

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "frame_visualizer"):
                self.frame_visualizer = VisualizationMarkers(
                    VisualizationMarkersCfg(
                        prim_path="/Visuals/SiteViz",
                        markers={
                            "sphere": sim_utils.SphereCfg(
                                        radius=0.05,
                                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                                        ),
                        }
                    )
                )
            # set their visibility to true
            self.frame_visualizer.set_visibility(False)
        else:
            if hasattr(self, "frame_visualizer"):
                self.frame_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Update the visualized markers
        if self.frame_visualizer is not None:
            self.frame_visualizer.visualize(self._data.root_pos_w.view(-1, 3), self._data.root_quat_w.view(-1, 4))


