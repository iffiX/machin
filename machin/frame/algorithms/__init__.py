import warnings
from .base import TorchFramework

from .dqn import DQN
from .dqn_per import DQNPer
from .rainbow import RAINBOW

from .ddpg import DDPG
from .hddpg import HDDPG
from .td3 import TD3
from .ddpg_per import DDPGPer

from .a2c import A2C
from .a3c import A3C
from .ppo import PPO
from .sac import SAC

from .maddpg import MADDPG

try:
    from .apex import DQNApex, DDPGApex
    from .impala import IMPALA
    from .ars import ARS
except ImportError as _:
    warnings.warn(
        "Failed to import algorithms relying on torch.distributed." " Set them to None."
    )
    DQNApex = None
    DDPGApex = None
    IMPALA = None
    ARS = None


__all__ = [
    "TorchFramework",
    "DQN",
    "DQNPer",
    "RAINBOW",
    "DDPG",
    "HDDPG",
    "TD3",
    "DDPGPer",
    "A2C",
    "A3C",
    "PPO",
    "SAC",
    "DQNApex",
    "DDPGApex",
    "IMPALA",
    "ARS",
    "MADDPG",
]
