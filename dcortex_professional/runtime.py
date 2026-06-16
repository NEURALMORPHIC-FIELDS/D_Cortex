# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Substrate LM runtime for the professional control layer. Wraps the frozen
# D_Cortex substrate (warmstarted_init.pt, GPT-2-medium warm-started decoder) and
# exposes (a) unconstrained greedy generation (what the raw model would say) and
# (b) CONSTRAINED generation where factual-slot tokens are mechanically forced to a
# committed value via logit masking. The model weights are read-only.

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import tiktoken

from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model

REPO_ROOT = Path(__file__).resolve().parent.parent
WARMSTART = REPO_ROOT.parent / "D_Cortex-main" / "runs" / "warmstart" / "warmstarted_init.pt"
ENC = tiktoken.get_encoding("gpt2")
NEG_INF = float("-inf")


def big_config() -> DCortexConfig:
    return DCortexConfig(hidden_dim=1024, n_enc_heads=16, n_dec_heads=16,
                         enc_ff_dim=4096, dec_ff_dim=4096, n_dec_layers=16,
                         n_enc_layers=4, n_fusion_layers=4, max_seq_len=2048)


@dataclass
class ConstrainedResult:
    text: str
    forced_value: str
    unconstrained_slot_text: str       # what the raw model wanted to emit at the slot
    overridden: bool                   # True if the constraint changed the emission


class SubstrateLM:
    """Frozen substrate language model with constrained decoding."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.available = False
        self.reason = ""
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[DCortexV2Model] = None
        if not WARMSTART.exists():
            self.reason = f"substrate checkpoint missing: {WARMSTART}"
            return
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ckpt = torch.load(WARMSTART, map_location=self.device, weights_only=False)
                model = DCortexV2Model(big_config()).to(self.device)
                model.load_state_dict(ckpt["model"])
                model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self.model = model
            self.available = True
        except Exception as exc:  # noqa: BLE001
            self.reason = f"substrate load failed: {type(exc).__name__}: {exc}"

    @torch.no_grad()
    def _next_logits(self, ids: List[int]) -> torch.Tensor:
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits = self.model.decode(x)            # [1, T, vocab]
        return logits[0, -1]                     # [vocab]

    @torch.no_grad()
    def generate_unconstrained(self, prompt: str, max_new_tokens: int = 12) -> str:
        """Greedy continuation from the raw model (may hallucinate)."""
        ids = ENC.encode_ordinary(prompt)
        out: List[int] = []
        for _ in range(max_new_tokens):
            nxt = int(self._next_logits(ids + out).argmax().item())
            out.append(nxt)
            piece = ENC.decode(out)
            if piece.endswith((".", "\n")) and len(out) >= 2:
                break
        return ENC.decode(out).strip()

    @torch.no_grad()
    def generate_constrained(self, prompt: str, forced_value: str) -> ConstrainedResult:
        """Emit `forced_value` at the factual slot by masking all other tokens.

        At each slot step the model's UNCONSTRAINED argmax is recorded for evidence,
        then the logits are masked so only the committed value's next token survives;
        the emission is therefore mechanically pinned to the committed value."""
        ids = ENC.encode_ordinary(prompt)
        value_ids = ENC.encode_ordinary((" " if not prompt.endswith(" ") else "") + forced_value)
        unconstrained_pieces: List[int] = []
        emitted: List[int] = []
        for tok in value_ids:
            logits = self._next_logits(ids + emitted)
            uncon = int(logits.argmax().item())
            unconstrained_pieces.append(uncon)
            # mechanical constraint: only the committed token may be emitted here
            mask = torch.full_like(logits, NEG_INF)
            mask[tok] = logits[tok]
            forced = int(mask.argmax().item())
            emitted.append(forced)
        unconstrained_slot = ENC.decode(unconstrained_pieces).strip()
        forced_text = ENC.decode(emitted).strip()
        overridden = ENC.decode(unconstrained_pieces).strip() != forced_text
        return ConstrainedResult(text=f"{prompt}{ENC.decode(emitted)}".strip(),
                                 forced_value=forced_value, unconstrained_slot_text=unconstrained_slot,
                                 overridden=overridden)

    @torch.no_grad()
    def hidden_states(self, text: str, max_seq_len: int = 128):
        """Frozen contextual token states for the neural binder path."""
        from dcortex.semantic_role_conditioned import DCortexTokenContextBackend
        backend = DCortexTokenContextBackend(self.model, lambda t: ENC.encode_ordinary(t),
                                             max_seq_len=max_seq_len)
        return backend
