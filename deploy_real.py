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
    env = RobotEnv(do_reset=False)
    breakpoint()
    # env.reset(randomize=False)

    # from Image
    from moviepy.editor import ImageSequenceClip
    import cv2
    import time
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
    video = []
    for i in range(100):
        t = time.time()
        obs = env.get_observation() 
        print(f"Observation Time: {time.time()-t}")

        images = obs["image"]
        combined_image = [v for k, v in images.items()]
        combined_image = np.concatenate(combined_image, axis=1)
        instruction = "pick up the pot"

        cv2.imshow("realsense_view", cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

        t = time.time()
        action = requests.post(
            "http://0.0.0.0:8000/act",
            json={"image": combined_image, 
                  "instruction": instruction,
                  "unnorm_key": "utaustin_mutex"}
        ).json()
        print(f"Pred Time: {time.time()-t}")

        print(action)
        action = action.copy()
        # action[-1] = -action[-1]
        action[-1] = ((1 - action[-1])-0.5)*2
        env.step(action)
        video.append(combined_image)
    
    # write video to disk
    ImageSequenceClip(video, fps=30).write_videofile(str("./data/latest_realworld.mp4"), codec="libx264", fps=10)

if __name__ == "__main__":
    # main()
    test()

