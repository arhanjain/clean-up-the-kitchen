import hydra
import h5py
import json
import random
import numpy as np
from cleanup.config import Config

from droid.droid.robot_env import RobotEnv

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: Config):
        
    # checks
    assert cfg.deploy.dataset != "", "Please provide a dataset"
    assert cfg.deploy.replay or cfg.deploy.checkpoint != "", "Please provide a checkpoint or set replay to True"

    env = RobotEnv()
    env.reset(randomize=False)

    hdf5  = h5py.File(cfg.deploy.dataset, "r")
    data = hdf5["data"]
    meta = json.loads(data.attrs["meta"])

    action_min = np.array(meta["action_min"])
    action_max = np.array(meta["action_max"])
    def unnormalize_action(action):
        # from -1,1 to min, max
        return (action + 1) * (action_max - action_min) / 2 + action_min
    
    if cfg.deploy.replay:
        demo = random.choice(list(data.keys()))
        data = data[demo]
        actions = data["actions"]
        print(f"Replaying data from {demo}")
        for i in range(actions.shape[0]):
            act = unnormalize_action(actions[i])
            print(f"Action: {act}")
            env.step(act)

        pass
    else:
        pass


def test():
    env = RobotEnv()
    env.reset(randomize=False)

    # from Image
    import cv2
    import requests
    import json_numpy
    json_numpy.patch()
    import numpy as np

    #
    # # Load Processor & VLA
    # processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    # vla = AutoModelForVision2Seq.from_pretrained(
    #     "openvla/openvla-7b", 
    #     attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    #     torch_dtype=torch.bfloat16, 
    #     low_cpu_mem_usage=True, 
    #     trust_remote_code=True
    # ).to("cuda:0")
    # 
    for i in range(100):
        obs = env.get_observation() 
        images = obs["image"]
        combined_image = [v for k, v in images.items()]
        combined_image = np.concatenate(combined_image, axis=1)
        instruction = "pick up the carrot"

        cv2.imshow("realsense_view", cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

        # Grab image input & format prompt
        # image = Image.fromarray(combined_image)
        # prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

        # Predict Action (7-DoF; un-normalize for BridgeData V2)
        # inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        # action = vla.predict_action(**inputs, unnorm_key="utaustin_mutex", do_sample=False)
        # gripper_act = obs["robot_state"]["gripper_position"]/0.08 
        action = requests.post(
            "http://0.0.0.0:8000/act",
            json={"image": combined_image, 
                  "instruction": instruction,
                  "unnorm_key": "utaustin_mutex"}
        ).json()
        print(action, type(action))
        # action = np.array(action)

    #     breakpoint()
    #     print(action)
    #     env.step(action)
    #

if __name__ == "__main__":
    # main()
    test()

