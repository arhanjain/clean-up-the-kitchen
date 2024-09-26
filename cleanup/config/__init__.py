from hydra.core.config_store import ConfigStore
from .config import Config

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
