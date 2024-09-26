from abc import abstractmethod
from robomimic.models import EncoderCore
from robomimic.utils.obs_utils import Modality

class RGBPCDModality(Modality):

    name = "rgb_pcd"

    @classmethod
    def _default_obs_processor(cls, obs):
        return obs
    
    @classmethod
    def _default_obs_unprocessor(cls, obs):
        return obs


