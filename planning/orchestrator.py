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

    def poop(self, plane_template):
        for action_type, object, location in plan_template:
            # rgb, seg, depth, meta_data = self.env.get_camera_data()
            # loaded_data = rgb, seg, depth, meta_data
            match action_type:
                case "move":
                    # get grasp pose
                    success = torch.zeros(self.env.num_envs)
                    while not torch.all(success):
                        grasp_pose, success = self.grasper.get_grasp(self.env, object)
                    pregrasp_pose = self.grasper.get_pregrasp(grasp_pose, 0.1)

                    # go to pregrasp
                    joint_pos, joint_vel, joint_names = self.env.get_joint_info()
                    traj, success = self.plan(joint_pos, joint_vel, joint_names, pregrasp_pose, mode="ee_pose")
                    if not success:
                        print("Failed to plan to pregrasp")
                        yield None
                    else:
                        traj, traj_length = self.test_format(traj, maxpad=max(t.ee_position.shape[0] for t in traj))
                        yield torch.cat((traj, torch.ones(self.env.num_envs, traj.shape[1], 1).to(self.device)), dim=2)
                    
                    
                    # go to grasp
                    action = torch.cat((grasp_pose, torch.ones(1,1)), dim=1).to(self.device)
                    yield action.repeat(1, 30, 1)
                    # joint_pos, joint_vel, joint_names = self.env.get_joint_info()
                    # traj, success = self.plan(joint_pos, joint_vel, joint_names, grasp_pose, mode="ee_pose")
                    # if not success:
                    #     print("Failed to plan to grasp")
                    #     yield None  
                    # else:
                    #     traj, traj_length = self.test_format(traj, maxpad=max(t.ee_position.shape[0] for t in traj))
                    #     # traj, traj_length = self.test_format(traj, maxpad=500)
                    #     yield torch.cat((traj, torch.ones(self.env.num_envs, traj.shape[1], 1).to(self.device)), dim=2)

                    # grasp
                    ee_frame_sensor = self.env.unwrapped.scene["ee_frame"]
                    tcp_rest_position = ee_frame_sensor.data.target_pos_source[..., 0, :].clone()
                    tcp_rest_orientation = ee_frame_sensor.data.target_quat_source[..., 0, :].clone()
                    ee_pose = torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1)
                    close_gripper = -1 * torch.ones(self.env.num_envs, 1).to(self.device)
                    yield torch.cat((ee_pose, close_gripper), dim=1).repeat(1, 20, 1)

                    # go to pregrasp
                    action = torch.cat((pregrasp_pose, -torch.ones(1,1)), dim=1).to(self.device)
                    yield action.repeat(1, 30, 1)
                    # joint_pos, joint_vel, joint_names = self.env.get_joint_info()
                    # traj, success = self.plan(joint_pos, joint_vel, joint_names, pregrasp_pose, mode="ee_pose")
                    # if not success:
                    #     print("Failed to plan to pregrasp")
                    #     yield None
                    # else:
                    #     traj, traj_length = self.test_format(traj, maxpad=max(t.ee_position.shape[0] for t in traj))
                    #     yield torch.cat((traj, -1*torch.ones(self.env.num_envs, traj.shape[1], 1).to(self.device)), dim=2)


                case _:
                    raise ValueError("Invalid action type!")
    
