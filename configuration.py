from datetime import datetime
from dataclasses import asdict, dataclass
from omni.isaac.lab.utils import configclass

@configclass
class GeneralCfg:
  usd_path: str = "/home/arhan/Downloads/bowlsite.usdz"
  log_dir: str = f"/home/arhan/projects/IsaacLab/source/standalone/clean-up-the-kitchen/logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


@configclass
class SB3Cfg:
  seed: int = 42

  n_timesteps: float = 1e7
  policy: str = 'MlpPolicy'
  batch_size: int = 128
  n_steps: int =  512
  gamma: float = 0.99
  gae_lambda: float = 0.9
  n_epochs: int = 20
  ent_coef: float = 0.0
  sde_sample_freq: int = 4
  max_grad_norm: float = 0.5
  vf_coef: float = 0.5
  learning_rate: float = 3e-5
  use_sde: bool =  True
  clip_range: float = 0.4
  device: str = "cuda:0"
  policy_kwargs: str = '''dict(
                            log_std_init=-1,
                            ortho_init=False,
                            activation_fn=nn.ReLU,
                            net_arch=dict(pi=[256, 256], vf=[256, 256])
                        )'''

  def dict(self):
    return asdict(self)


# default:
#   usd_file: "test.usdz"

# task: None
# num_envs: 2