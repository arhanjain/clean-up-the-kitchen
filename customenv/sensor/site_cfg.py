from omni.isaac.lab.sensors import SensorBaseCfg
from omni.isaac.lab.utils import configclass
from typing import Tuple, TYPE_CHECKING

from .site import Site

@configclass
class SiteCfg(SensorBaseCfg):
    class_type: type = Site

    offset: Tuple[float, float, float] = (0,0,0)