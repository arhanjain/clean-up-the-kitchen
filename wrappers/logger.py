import dm_env
import envlogger
import tensorflow_datasets as tfds
import gymnasium as gym
import numpy as np
import tensorflow as tf

from gymnasium import spaces
from dm_env import specs
from typing import Optional, Dict
from envlogger.backends.tfds_backend_writer import TFDSBackendWriter

def space2spec(space: gym.Space, name: Optional[str] = None):
  """Converts an OpenAI Gym space to a dm_env spec or nested structure of specs.

  Box, MultiBinary and MultiDiscrete Gym spaces are converted to BoundedArray
  specs. Discrete OpenAI spaces are converted to DiscreteArray specs. Tuple and
  Dict spaces are recursively converted to tuples and dictionaries of specs.

  Args:
    space: The Gym space to convert.
    name: Optional name to apply to all return spec(s).

  Returns:
    A dm_env spec or nested structure of specs, corresponding to the input
    space.
  """
  if isinstance(space, spaces.Discrete):
    return specs.DiscreteArray(num_values=space.n, dtype=space.dtype, name=name)

  elif isinstance(space, spaces.Box):
    return specs.BoundedArray(shape=space.shape, dtype=space.dtype, minimum=space.low, maximum=space.high, name=name)

  elif isinstance(space, spaces.MultiBinary):
    return specs.BoundedArray(shape=space.shape, dtype=space.dtype, minimum=0.0,
                              maximum=1.0, name=name)

  elif isinstance(space, spaces.MultiDiscrete):
    return specs.BoundedArray(shape=space.shape, dtype=space.dtype,
                              minimum=np.zeros(space.shape),
                              maximum=space.nvec, name=name)

  elif isinstance(space, spaces.Tuple): return tuple(space2spec(s, name) for s in space.spaces)
  elif isinstance(space, spaces.Dict):
    return {key: space2spec(value, name) for key, value in space.spaces.items()}

  else:
    raise ValueError('Unexpected gym space: {}'.format(space))


class DMEnvFromGym(dm_env.Environment):
    """A wrapper to convert an OpenAI Gym environment to a dm_env.Environment."""

    def __init__(self, gym_env: gym.Env):
        self.gym_env = gym_env
        # Convert gym action and observation spaces to dm_env specs.
        self._observation_spec = space2spec(self.gym_env.observation_space,
                                            name='observations')
        self._action_spec = space2spec(self.gym_env.action_space, name='actions')
        self._reset_next_step = True

    def reset(self) -> dm_env.TimeStep:
        self._reset_next_step = False
        observation, info = self.gym_env.reset()
        observation = torchdict2numpydict(observation)
        return dm_env.restart(observation)

    def step(self, action: int) -> dm_env.TimeStep:
        if self._reset_next_step:
          return self.reset()

        # Convert the gym step result to a dm_env TimeStep.
        observation, reward, term, trunc, info = self.gym_env.step(action)
        observation = torchdict2numpydict(observation)
        self._reset_next_step = term or trunc

        if term:
            return dm_env.termination(reward, observation)
        elif trunc:
            return dm_env.truncation(reward, observation)

        return dm_env.transition(reward, observation)

    def close(self):
        self.gym_env.close()

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    @property
    def unwrapped(self):
        return self.gym_env

def torchdict2numpydict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = torchdict2numpydict(v)
        else:
            d[k] = v.cpu().numpy()
    return d

def spec2tensors(spec: Dict):
    features = {}
    for k, v in spec.items():
        if isinstance(v, dict):
            features[k] = spec2tensors(v)
        else:
            features[k] = tfds.features.Tensor(
                    shape=v.shape,
                    dtype=v.dtype,
                    )
    return features

def wrap_env_in_logger(env):

    dm_wrapped_env = DMEnvFromGym(env)
    
    obs_dict = spec2tensors(dm_wrapped_env.observation_spec())
    obs_info = tfds.features.FeaturesDict(obs_dict)

    # action_dict = spec2tensors(dm_wrapped_env.action_spec())
    # action_info = tfds.features.FeaturesDict(action_dict)
    action_spec = dm_wrapped_env.action_spec()
    action_info = tfds.features.Tensor(
            shape=action_spec.shape,
            dtype=action_spec.dtype,
            )


    dataset_config = tfds.rlds.rlds_base.DatasetConfig(
            name = "test_ds",
            observation_info = obs_info,
            action_info = action_info,
            reward_info = tfds.features.Scalar(dtype=tf.float32),
            discount_info = tfds.features.Scalar(dtype=tf.float32),
            )
    logger = envlogger.EnvLogger(
            dm_wrapped_env,
            backend = TFDSBackendWriter(
                    data_directory = "./data/ds",
                    split_name = "train",
                    max_episodes_per_file=50,
                    ds_config = dataset_config,
                )
            )
    return logger


