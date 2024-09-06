# from .grasp import Grasper
# from .motion_planner import MotionPlanner
from .actions import Action, ServiceName
import torch
from openai import OpenAI
from config import Config
import yaml
import re
from transformers import AutoModelForVision2Seq, AutoProcessor

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
        self.env = env
        # self.grasper = Grasper(
        #         env,
        #         cfg.grasp,
        #         cfg.usd_path,
        #         )
        # self.motion_planner = MotionPlanner(env)
        
        # openvla_processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        # openvla = AutoModelForVision2Seq.from_pretrained(
        #     "openvla/openvla-7b", 
        #     attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        #     torch_dtype=torch.bfloat16, 
        #     low_cpu_mem_usage=True, 
        #     trust_remote_code=True
        # ).to("cuda:0")
        # Action.register_service(ServiceName.GRASPER, self.grasper)
        # Action.register_service(ServiceName.MOTION_PLANNER, self.motion_planner)
        # Action.register_service(ServiceName.OPEN_VLA, (openvla, openvla_processor))

    
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

    def generate_cleanup_tasks(scene_info):
        """
        Generates a list of tasks needed to clean up the kitchen based on the scene layout.

        Args:
        - scene_info (list): A list of objects in the scene.

        Returns:
        - str: Formatted string with tasks for cleaning up the kitchen.
        """

        prompt_template = """Objective: 
        Given a scene layout, generate a list of tasks needed to clean up the kitchen using actions like "grasp" and "place". Each task should be in the format of a ("action_type", {{"target": "object_name"}}). 
        Identify objects that should be picked up and specify their target object.

        Task: Clean up the kitchen
        Scene: {scene}

        Examples:

        1. Task: Clean up the kitchen
        Scene: ['cup', 'table']
        Answer: [('grasp', {{"target": "cup"}}), ('place', {{"target": "table"}})]

        2. Task: Clean up the kitchen
        Scene: ['plate', 'cabinet']
        Answer: [('grasp', {{"target": "plate"}}), ('place', {{"target": "cabinet"}})]

        3. Task: Clean up the kitchen
        Scene: ['bowl', 'cup', 'table', 'cabinet']
        Answer: [('grasp', {{"target": "bowl"}}), ('place', {{"target": "cabinet"}}), ('grasp', {{"target": "cup"}}), ('place', {{"target": "cabinet"}})]

        4. Task: Clean up the kitchen
        Scene: ['cup', 'sink', 'cabinet']
        Answer: [('grasp', {{"target": "cup"}}), ('place', {{"target": "sink"}})]

        5. Task: Clean up the kitchen
        Scene: ['bowl', 'sink', 'cup', 'plate', 'cabinet', 'table']
        Answer: [('grasp', {{"target": "bowl"}}), ('place', {{"target": "sink"}}), ('grasp', {{"target": "cup"}}), ('place', {{"target": "cabinet"}}), ('grasp', {{"target": "plate"}}), ('place', {{"target": "cabinet"}})]

        Generate ONLY the answer for the question above (e.g, [('grasp', {{"target": "bowl"}}), ('place', {{"target": "sink"}})]):
        """
        
        # Format scene_info as a string that can be inserted into the prompt
        scene_info_str = str(scene_info)

        # Apply the template formatting
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

    def parse_plan_template(plan_str):
        # # Remove comments and strip the string
        # plan_str = re.sub(r'#.*', '', plan_str).strip()

        # # Define regex patterns to extract actions and parameters
        # action_pattern = re.compile(r'\(\s*\'(\w+)\'\s*,\s*{([^}]*)}\s*\)')
        # param_pattern = re.compile(r'\'(\w+)\'\s*:\s*\'(\w+)\'')

        # plan = []
        # for action_match in action_pattern.finditer(plan_str):
        #     action = action_match.group(1)
        #     params_str = action_match.group(2)
        #     params = {}
            
        #     for param_match in param_pattern.finditer(params_str):
        #         key = param_match.group(1)
        #         value = param_match.group(2)
        #         params[key] = value
            
        #     plan.append((action, params))

        # return plan
        pass
