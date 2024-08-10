from .grasp import Grasper
from .motion_planner import MotionPlanner
from .actions import Action, ServiceName
import torch
from openai import OpenAI
import yaml

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

    # Can be used to understand
    def execute_plan(self, plan_template):
        '''
        Executes the action plan based on the provided plan template.

        Parameters
        ----------
        plan_template: list
            A list of action tuples, where each tuple contains the action type, 
            object, and location.

        Yields
        ------
        torch.Tensor((1, 8), dtype=torch.float32)
            The next action to be taken by the robot.
        '''
        for action_type, obj, location in plan_template:
            match action_type:
                case "move":
                    # Get the best manipulation pose for the object
                    success = torch.zeros(self.env.num_envs)
                    while not torch.all(success):
                        manipulation_pose, success = self.grasper.get_manipulation(self.env, obj, 'grasps')
                    pre_manipulation_pose = self.grasper.get_pregrasp(manipulation_pose, 0.1)

                    # Move to the pre-manipulation position
                    joint_pos, joint_vel, joint_names = self.env.get_joint_info()
                    traj, success = self.motion_planner.plan_path(joint_pos, joint_vel, joint_names, pre_manipulation_pose, mode="ee_pose")
                    if not success:
                        print("Failed to plan to pre-manipulation position")
                        yield None
                    else:
                        traj, traj_length = self.format_trajectory(traj, maxpad=max(t.ee_position.shape[0] for t in traj))
                        yield torch.cat((traj, torch.ones(self.env.num_envs, traj.shape[1], 1).to(self.env.unwrapped.device)), dim=2)

                    # Move to the manipulation position
                    action = torch.cat((manipulation_pose, torch.ones(1, 1)), dim=1).to(self.env.unwrapped.device)
                    yield action.repeat(1, 30, 1)

                    # Execute the grasp (close the gripper)
                    ee_frame_sensor = self.env.unwrapped.scene["ee_frame"]
                    tcp_rest_position = ee_frame_sensor.data.target_pos_source[..., 0, :].clone()
                    tcp_rest_orientation = ee_frame_sensor.data.target_quat_source[..., 0, :].clone()
                    ee_pose = torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1)
                    close_gripper = -1 * torch.ones(self.env.num_envs, 1).to(self.env.unwrapped.device)
                    yield torch.cat((ee_pose, close_gripper), dim=1).repeat(1, 20, 1)

                    # Move back to the pre-manipulation position
                    action = torch.cat((pre_manipulation_pose, -torch.ones(1, 1)), dim=1).to(self.env.unwrapped.device)
                    yield action.repeat(1, 30, 1)
                    joint_pos, joint_vel, joint_names = self.env.get_joint_info()
                    traj, success = self.motion_planner.plan_path(joint_pos, joint_vel, joint_names, pre_manipulation_pose, mode="ee_pose")
                    if not success:
                        print("Failed to plan back to pre-manipulation position")
                        yield None
                    else:
                        traj, traj_length = self.format_trajectory(traj, maxpad=max(t.ee_position.shape[0] for t in traj))
                        yield torch.cat((traj, -1 * torch.ones(self.env.num_envs, traj.shape[1], 1).to(self.env.unwrapped.device)), dim=2)

                    # Move to the place location
                    place_pose, place_success = self.grasper.get_manipulation(self.env, obj, 'placements')
                    joint_pos, joint_vel, joint_names = self.env.get_joint_info()
                    traj, place_success = self.motion_planner.plan_path(joint_pos, joint_vel, joint_names, place_pose, mode="ee_pose")
                    if not place_success:
                        print("Failed to plan to placement location")
                        yield None
                    else:
                        traj, traj_length = self.format_trajectory(traj, maxpad=max(t.ee_position.shape[0] for t in traj))
                        yield torch.cat((traj, torch.ones(self.env.num_envs, traj.shape[1], 1).to(self.env.unwrapped.device)), dim=2)

                    # Execute the place action (move to place pose)
                    action = torch.cat((place_pose, torch.ones(1, 1)), dim=1).to(self.env.unwrapped.device)
                    yield action.repeat(1, 30, 1)

                    # Open the gripper to release the object
                    ee_frame_sensor = self.env.unwrapped.scene["ee_frame"]
                    tcp_rest_position = ee_frame_sensor.data.target_pos_source[..., 0, :].clone()
                    tcp_rest_orientation = ee_frame_sensor.data.target_quat_source[..., 0, :].clone()
                    ee_pose = torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1)
                    open_gripper = torch.ones(self.env.num_envs, 1).to(self.env.unwrapped.device)
                    yield torch.cat((ee_pose, open_gripper), dim=1).repeat(1, 20, 1)

                case _:
                    raise ValueError("Invalid action type!")

    
    def generate_cleanup_tasks(scene_info):
        """
        Generates a list of tasks needed to clean up the kitchen based on the scene layout.

        Args:
        - scene_info (list): A list of objects in the scene.

        Returns:
        - str: Formatted string with tasks for cleaning up the kitchen.
        """

        prompt_template = """Objective: 
        Given a scene layout, generate a list of tasks needed to clean up the kitchen using actions like "grasp" and "place". Each task should be in the format of a ("action_type", {"target": "object_name"}). 
        Identify objects that should be picked up and specify their target object.

        Task: Clean up the kitchen
        Scene: {scene}

        Examples:

        1. Task: Clean up the kitchen
        Scene: ['cup', 'table']
        Answer: [('grasp', {'target': 'cup'}), ('place', {'target': 'table'})]

        2. Task: Clean up the kitchen
        Scene: ['plate', 'cabinet']
        Answer: [('grasp', {'target': 'plate'}), ('place', {'target': 'cabinet'})]

        3. Task: Clean up the kitchen
        Scene: ['bowl', 'cup', 'table', 'cabinet']
        Answer: [('grasp', {'target': 'bowl'}), ('place', {'target': 'cabinet'}), ('grasp', {'target': 'cup'}), ('place', {'target': 'cabinet'})]

        4. Task: Clean up the kitchen
        Scene: ['cup', 'sink', 'cabinet']
        Answer: [('grasp', {'target': 'cup'}), ('place', {'target': 'sink'})]

        5. Task: Clean up the kitchen
        Scene: ['bowl', 'sink', 'cup', 'plate', 'cabinet', 'table']
        Answer: [('grasp', {'target': 'bowl'}), ('place', {'target': 'sink'}), ('grasp', {'target': 'cup'}), ('place', {'target': 'cabinet'}), ('grasp', {'target': 'plate'}), ('place', {'target': 'cabinet'})]

        Generate ONLY the answer for the question above (e.g, [('grasp', {'target': 'bowl'}), ('place', {'target': 'sink'})]):
        """
        
        # Format scene_info as a string that can be inserted into the prompt
        scene_info_str = str(scene_info)

        formatted_prompt = prompt_template.format(scene=scene_info_str)
        
        return formatted_prompt


    def extract_objects_and_sites_info(usd_info_path):
        with open(usd_info_path) as f:
            usd_info = yaml.safe_load(f)
        scene_info = []
        for obj_name, _ in usd_info["xforms"].items():
            scene_info.append(obj_name)
        return scene_info

    def get_plan(prompt):
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content
