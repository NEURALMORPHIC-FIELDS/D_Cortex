"""D_Cortex v2.0-alpha -- dual-agent memory-native transformer."""

from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
from dcortex.encoder import MemoryEncoder

__all__ = ["DCortexConfig", "DCortexV2Model", "MemoryEncoder"]
__version__ = "2.0.0-alpha"
