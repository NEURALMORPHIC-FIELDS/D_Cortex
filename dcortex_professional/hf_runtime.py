# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Real open base-model runtime for the professional control layer. Wraps a Hugging
# Face causal LM (default gpt2-large, fp32, full hidden states + logits) and exposes
# (a) unconstrained greedy generation (the RAW model, which hallucinates plausibly),
# (b) CONSTRAINED generation where factual-slot tokens are mechanically forced to a
# committed value via logit masking, and (c) span-pooled hidden states for the
# neural binder. Same interface as the substrate runtime so the control organism is
# model-agnostic. Model weights are read-only.

import contextlib
import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class ConstrainedResult:
    text: str
    forced_value: str
    unconstrained_slot_text: str
    overridden: bool


class HFBaseModel:
    """Real open-weights causal LM with constrained decoding and hidden states."""

    def __init__(self, model_name: str = "gpt2-large", device: Optional[str] = None,
                 fallback: str = "gpt2-medium") -> None:
        self.available = False
        self.reason = ""
        self.model_name = model_name
        self.precision = "fp32"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.tok = None
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            self.reason = f"transformers unavailable: {exc}"
            return
        for name in (model_name, fallback):
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    tok = AutoTokenizer.from_pretrained(name)
                    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
                    model.to(self.device).eval()
                for p in model.parameters():
                    p.requires_grad_(False)
                self.model, self.tok, self.model_name = model, tok, name
                self.available = True
                break
            except Exception as exc:  # noqa: BLE001
                self.reason = f"load {name} failed: {type(exc).__name__}: {exc}"

    @torch.no_grad()
    def _next_logits(self, ids: List[int]) -> torch.Tensor:
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        return self.model(x).logits[0, -1]

    @torch.no_grad()
    def generate_unconstrained(self, prompt: str, max_new_tokens: int = 16) -> str:
        ids = self.tok.encode(prompt)
        out: List[int] = []
        for _ in range(max_new_tokens):
            nxt = int(self._next_logits(ids + out).argmax().item())
            out.append(nxt)
            piece = self.tok.decode(out)
            if piece.endswith((".", "\n")) and len(out) >= 2:
                break
        return self.tok.decode(out).strip()

    @torch.no_grad()
    def generate_constrained(self, prompt: str, forced_value: str) -> ConstrainedResult:
        ids = self.tok.encode(prompt)
        value_ids = self.tok.encode((" " if not prompt.endswith(" ") else "") + forced_value)
        unconstrained: List[int] = []
        emitted: List[int] = []
        for tok_id in value_ids:
            logits = self._next_logits(ids + emitted)
            unconstrained.append(int(logits.argmax().item()))
            emitted.append(tok_id)        # mechanical constraint: only the committed token survives
        forced_text = self.tok.decode(emitted).strip()
        uncon_text = self.tok.decode(unconstrained).strip()
        return ConstrainedResult(text=f"{prompt}{self.tok.decode(emitted)}".strip(),
                                 forced_value=forced_value, unconstrained_slot_text=uncon_text,
                                 overridden=uncon_text != forced_text)

    @torch.no_grad()
    def span_features(self, text: str, phrases: List[str]) -> Optional[torch.Tensor]:
        """Mean-pooled last-layer hidden state for each phrase span. Returns
        [len(phrases), hidden] or None if any phrase is not locatable."""
        enc = self.tok(text, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        hidden = self.model(**enc, output_hidden_states=True).hidden_states[-1][0]  # [seq, dim]
        low = text.lower()
        pooled = []
        for phrase in phrases:
            start = low.find(phrase.lower())
            if start < 0:
                return None
            end = start + len(phrase)
            idx = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
            if not idx:
                return None
            pooled.append(hidden[idx].mean(dim=0))
        return torch.stack(pooled, dim=0)

    @property
    def hidden_dim(self) -> int:
        return int(self.model.config.hidden_size) if self.available else 0
