import carb
import typing

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils

from pxr import Usd, Sdf
from collections.abc import Callable
from omni.isaac.lab.sim import schemas
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.sim.utils import clone, select_usd_variants, bind_visual_material

def add_reference_to_stage_custom(usd_path: str, usd_sub_path:str, prim_path: str, prim_type: str = "Xform") -> Usd.Prim:
    stage = prim_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim = stage.DefinePrim(prim_path, prim_type)
    carb.log_info("Loading Asset from path {} ".format(usd_path))
    success_bool = prim.GetReferences().AddReference(
        usd_path,
        Sdf.Path(usd_sub_path)
    )
    if not success_bool:
        raise FileNotFoundError("The usd file at path {} provided wasn't found".format(usd_path))
    return prim


def create_prim_custom(
    prim_path: str,
    prim_type: str = "Xform",
    position: typing.Optional[typing.Sequence[float]] = None,
    translation: typing.Optional[typing.Sequence[float]] = None,
    orientation: typing.Optional[typing.Sequence[float]] = None,
    scale: typing.Optional[typing.Sequence[float]] = None,
    usd_path: typing.Optional[str] = None,
    usd_sub_path = None,
    semantic_label: typing.Optional[str] = None,
    semantic_type: str = "class",
    attributes: typing.Optional[dict] = None,
) -> Usd.Prim:
    # Note: Imported here to prevent cyclic dependency in the module.
    from omni.isaac.core.prims.xform_prim import XFormPrim

    # create prim in stage
    prim = prim_utils.define_prim(prim_path=prim_path, prim_type=prim_type)
    if not prim:
        return None
    # apply attributes into prim
    if attributes is not None:
        for k, v in attributes.items():
            prim.GetAttribute(k).Set(v)
    # add reference to USD file
    if usd_path is not None:
        add_reference_to_stage_custom(
            usd_path=usd_path,
            usd_sub_path=usd_sub_path,
            prim_path=prim_path
        )
    # add semantic label to prim
    if semantic_label is not None:
        prim_utils.add_update_semantics(prim, semantic_label, semantic_type)
    # apply the transformations
    XFormPrim(prim_path=prim_path, position=position, translation=translation, orientation=orientation, scale=scale)

    return prim



@clone
def spawn_from_usd_test(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    # spawn asset from the given usd file
    # return _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    stage: Usd.Stage = stage_utils.get_current_stage()
    if not stage.ResolveIdentifierToEditTarget(cfg.usd_path):
        raise FileNotFoundError(f"USD file not found at path: '{cfg.usd_path}'.")
    # spawn asset if it doesn't exist.
    if not prim_utils.is_prim_path_valid(prim_path):
        # add prim as reference to stage
        create_prim_custom(
            prim_path,
            usd_path=cfg.usd_path,
            translation=translation,
            orientation=orientation,
            scale=cfg.scale,
            usd_sub_path=cfg.usd_sub_path
        )
        # prim_utils.create_prim(
        #     prim_path,
        #     usd_path=cfg.usd_path,
        #     translation=translation,
        #     orientation=orientation,
        #     scale=cfg.scale,
        # )
    else:
        carb.log_warn(f"A prim already exists at prim path: '{prim_path}'.")

    # modify variants
    if hasattr(cfg, "variants") and cfg.variants is not None:
        select_usd_variants(prim_path, cfg.variants)

    # objects = {}
    # joints = {}
    prim = prim_utils.get_prim_at_path(prim_path)
    prim_children = prim_utils.get_prim_children(prim)
    for child in prim_children:
        # child = prim_utils.get_prim_children(child)[0]
        path = prim_utils.get_prim_path(child)
        name = path.split("/")[-1]

        print(name)
        print(prim_utils.get_prim_object_type(path))
        if prim_utils.get_prim_object_type(path) == "rigid_body":
            # TODO: path here should be with a * in env_0
            # path = path.replace("env_0","*")
            print(path)
            schemas.define_rigid_body_properties(path, cfg.rigid_props)
            schemas.define_collision_properties(path, cfg.collision_props)

            grandsons = prim_utils.get_prim_children(child)
            for grandson in grandsons:
                grandson_path = prim_utils.get_prim_path(grandson)
                grandson_name = grandson_path.split("/")[-1]
                if grandson_name == "FixedJoint":
                    print("Next, this is a fixed joint", grandson_name)
                    continue

                grandson_name = name+"/"+grandson_name
                print(grandson_name)
                print(prim_utils.get_prim_object_type(grandson_path))

    # return the prim
    return prim_utils.get_prim_at_path(prim_path)


@configclass
class CustomRigidUSDCfg(UsdFileCfg):
    usd_sub_path: str | None = None
    func: Callable = spawn_from_usd_test



#BACKUP
    def get_camera_data(self):
        # RGB Image
        rgb = self.scene["camera"].data.output["rgb"]

        # Mask 
        seg = self.scene["camera"].data.output["semantic_segmentation"]
        mask = torch.clamp(seg-1, max=1).cpu().numpy().astype(np.uint8) * 255

        # Depth values per pixel
        depth = self.scene["camera"].data.output["distance_to_image_plane"]

        # Assemble metadata
        metadata = {}

        # WARNING, its batched by the number of environments

        # save_dir = "/home/arhan/projects/IsaacLab/source/standalone/clean-up-the-kitchen/data/"
        # save rgb
        # Image.fromarray(rgb[0].cpu().numpy()).convert("RGB").save(f"{save_dir}/rgb.png")
        # # save depth
        # np.save(f"{save_dir}/depth.npy", depth[0])
        # # save seg
        # mask = torch.clamp(seg-1, max = 1).cpu().numpy().astype(np.uint8) * 255
        # Image.fromarray(mask[0], mode="L").save(f"{save_dir}/seg.png")

        # metadata
        metadata = {}
        intrinsics = self.scene["camera"].data.intrinsic_matrices[0]

        # camera pose
        cam_pos = self.scene["camera"].data.pos_w
        cam_quat = self.scene["camera"].data.quat_w_ros

        robot_pos = self.scene["robot"].data.root_state_w[:, :3]
        robot_quat = self.scene["robot"].data.root_state_w[:, 3:7]

        cam_pos_r, cam_quat_r = math.subtract_frame_transforms(
            robot_pos, robot_quat,
            cam_pos, cam_quat
        )
        cam_rot_mat_r = math.matrix_from_quat(cam_quat_r)
        cam_pos_r = cam_pos_r.unsqueeze(2)

        transformation = torch.cat((cam_rot_mat_r, cam_pos_r), dim=2).cpu()

        bottom_row = torch.tensor([0,0,0,1]).expand(self.num_envs, 1, 4)
        transformation = torch.cat((transformation, bottom_row), dim=1).numpy()


        # filler from existing file
        ee_pose = np.array([[ 0.02123945,  0.82657526,  0.56242531,  0.18838109],
        [ 0.99974109, -0.02215279, -0.00519713, -0.01743025],
        [ 0.00816347,  0.56239007, -0.82683176,  0.6148137 ],
        [ 0.        ,  0.        ,  0.        ,  1.        ]])
        scene_bounds = np.array([-0.4, -0.8, -0.2, 1.2, 0.8, 0.6])

        metadata["intrinsics"] = intrinsics.cpu().numpy()
        metadata["camera_pose"] = transformation[0]
        metadata["ee_pose"] = ee_pose
        metadata["label_map"] = None
        # metadata["scene_bounds"] = scene_bounds

        # with open(f"{save_dir}/meta_data.pkl", "wb") as f:
        #     pickle.dump(metadata, f)

        return rgb, seg, depth