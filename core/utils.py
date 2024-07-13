"""Utility module"""

import yaml
from easydict import EasyDict


############################################################
# Singleton manager
############################################################
class SingletonManager:
    """Singleton manager
    This class assures that only one instance is created.

    Attributes:
        instance: Singleton instance
    """

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(SingletonManager, cls).__new__(cls)
        return cls.instance

SINGLETON_MANAGER = SingletonManager()


############################################################
# File managing
############################################################
def load_yaml(path: str) -> EasyDict:
    """Load yaml file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return EasyDict(config)