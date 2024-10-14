from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
import numpy as np

class PlanGenerator:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(
            'allenai/Molmo-7B-O-0924',
            trust_remote_code=True,
            device_map='auto'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            'allenai/Molmo-7B-O-0924',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto'
        )


    def generate_plan(self, scene_info):
        prompt = self.construct_prompt(scene_info)
        plan_str = self.run_molmo(prompt)
        plan = self.parse_plan(plan_str)
        return plan

    def construct_prompt(self, scene_info):
        prompt = (
            f"Given a kitchen scene with the following objects: {', '.join(scene_info)}. "
            "Generate a sequence of actions to clean up the kitchen. Use actions like "
            "'grasp' and 'place' in the format ('action_type', {'target': 'object_name'}). "
            "Provide the sequence of actions needed to clean up.\n\n"
            "Answer:"
        )
        return prompt

    def run_molmo(self, prompt):
        rgb_image, _, _, _ = self.env.get_camera_data()
        img = Image.fromarray(rgb_image.squeeze().astype(np.uint8))
        breakpoint()
        inputs = self.processor.process(images=[img], text=prompt)
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            inputs["images"] = inputs["images"].to(torch.bfloat16)
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer
            )

        # Decode the generated plan
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        plan_str = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return plan_str.strip()

    def parse_plan(self, plan_str):
        # Hardcoded for testing
        plan_template = [
            ("grasp", {"target": "spoon"}),
            ("place", {"target": "spoon"}),           
            ("grasp", {"target": "cup"}),
            ("place", {"target": "cup"}),
            ("grasp", {"target": "ketchup"}),
            ("place", {"target": "ketchup"}),
        ]
        return plan_template
        import ast
        try:
            plan = ast.literal_eval(plan_str)
            if not isinstance(plan, list):
                raise ValueError("Generated plan is not a list.")
            return plan
        except Exception as e:
            print(f"Failed to parse plan: {e}")
            return []
