from .grasp import Grasper
from .motion_planner import MotionPlanner
from .actions import Action, ServiceName

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
    def __init__(self, env, cfg):
        self.env = env
        self.grasper = Grasper(
                cfg.grasp,
                cfg.usd_info.usd_path,
                self.env.unwrapped.device
                )
        self.motion_planner = MotionPlanner(env)

        Action.register_service(ServiceName.GRASPER, self.grasper)
        Action.register_service(ServiceName.MOTION_PLANNER, self.motion_planner)

    
    def generate_plan_from_template(self, plan_template):
        '''
        Computes the action sequence for a given plan template.

        Parameters
        ----------
        plan_template: list
            A list of ActionType objects that specify the actions 
            to be taken by the robot.

        Yields
        ------
        torch.Tensor((1, 8), dtype=torch.float32)
            The next action to be taken by the robot.
        '''
        for action, action_kwargs in plan_template:
            action = Action.create(action, **action_kwargs)
            for step in action.build(self.env):
                yield step

