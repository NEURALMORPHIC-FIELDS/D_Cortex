"""D_Cortex v2.0-alpha (sealed milestone v15.7a, 2026-04-26).

Dual-agent memory-native transformer with longitudinal consolidation.

The ``dcortex`` package exposes the v11 substrate (encoder/decoder/banks).
The v15.x layer (Pas 6 Role-of-Modifier Resolver + Pas 7a consolidator
pipeline at end_episode: reconcile -> prune -> retrograde -> promote) is
delivered as a sealed monolithic source in
``steps/13_v15_7a_consolidation/code.py``. See
``paper/D_CORTEX_PAS7A_SEAL.md`` for the citable seal certificate.
"""

from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
from dcortex.encoder import MemoryEncoder

__all__ = ["DCortexConfig", "DCortexV2Model", "MemoryEncoder"]

# Substrate (foundational v11) version
__version__ = "2.0.0-alpha"

# Current sealed milestone (full pipeline including v15.x consolidator)
__milestone__ = "v15.7a"
__milestone_date__ = "2026-04-26"
__milestone_artifact__ = "paper/D_CORTEX_PAS7A_SEAL.md"
