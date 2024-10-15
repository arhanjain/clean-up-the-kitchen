from .motion_planner import MotionPlanner
from .actions import Action, ServiceName
# from .plan_generator import PlanGenerator
from cleanup.config import Config
from .grasp import Grasper
from pxr import Usd, UsdGeom
import torch

class Orchestrator:
    '''
    Manages the interplay between several systems (motion planning, grasp prediction,
    policy rollouts) to produce robot trajectories.
    
    Parameters
    ----------
    env: gym.Env
        The environment to plan in.
    cfg: OmegaConf
        The configuration of the system.

    Attributes
    ----------
    env: gym.Env
        The environment to plan in.
    grasper: Grasper
        The grasp prediction system.
    planner: MotionPlanner
        The motion planning system.
    
    '''
    def __init__(self, env, cfg: Config):
        self.cfg = cfg
        self.env = env
        self.grasper = Grasper(
                env,
                cfg.grasp,
                cfg.usd_path,
                )
        self.motion_planner = MotionPlanner(env)
        # self.plan_generator = PlanGenerator(env, cfg)
        
        # Register services
        Action.register_service(ServiceName.GRASPER, self.grasper)
        Action.register_service(ServiceName.MOTION_PLANNER, self.motion_planner)

    def run(self):
        # scene_info = self.extract_scene_info()
        # breakpoint()
        # plan_template = self.plan_generator.generate_plan(scene_info)
        # if not plan_template:
        #     print("No valid plan generated.")
        #     return
        plan_template = [
            ("open_drawer", {}),
            # ("grasp", {"target": "spoon"}),
            # ("open_drawer", {}),           
            # ("grasp", {"target": 'mustard_bottle'}),
            # ("place", {"target": "cup"}),
            # ("grasp", {"target": "ketchup"}),
            # ("place", {"target": "ketchup"}),
        ]
        for action_name, action_kwargs in plan_template:
            action = Action.create(action_name, **action_kwargs)
            for step in action.build(self.env):
                yield step

    def extract_scene_info(self):
        """
        Extracts objects from the USD file specified by usd_path.

        Returns:
        --------
        list: A list of object names in the scene.
        """
        stage = Usd.Stage.Open(self.cfg.usd_path)
        
        if not stage:
            print(f"Unable to open USD file at {self.grasper.usd_path}")
            return []

        scene_info = []
        
        root_prim_path = '/World/envs/env_0/'
        root_prim = stage.GetPrimAtPath(root_prim_path)
        if not root_prim:
            print(f"No '{root_prim_path}' prim found in the stage.")
            return scene_info
        
        for prim in root_prim.GetAllChildren():
            if prim.GetTypeName() in ['Xform', 'Mesh']:
                obj_name = prim.GetName()
                if obj_name not in ['GroundPlane', 'Camera', 'Lights']:
                    scene_info.append(obj_name)
        return scene_info
