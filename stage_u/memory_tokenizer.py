# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Memory tokenizer (the learned codebook): turns the model's INTERNALIZED value vector (w_value,
# Step B) into a discrete, scalable HANDLE - so the proven Stage U memory can hold a real domain.
# Same value -> same token (the exact same/different identity the honest mechanics need); distinct
# value -> distinct token; the codebook holds K slots (K >> the sealed organ's 37). This is the
# gold-ANCHORED prototype form (codebook entry per known value = its mean internalized vector); a
# full VQ-VAE training is the later refinement. The token derives from the INTERNALIZED value, NOT
# from the value's text token - the line that keeps the axis-inversion (not regressing to copying).

from typing import Dict, List, Optional

import torch


class MemoryTokenizer:
    def __init__(self, capacity: int = 512, dim: int = 768) -> None:
        self.capacity = capacity
        self.dim = dim
        self.codebook = torch.zeros(0, dim)     # [n_used, dim], unit rows
        self.token_value: List[str] = []        # token id -> value name (the codebook<->value table)
        self.value_token: Dict[str, int] = {}   # value name -> token id

    def _unit(self, v: torch.Tensor) -> torch.Tensor:
        return v / (v.norm() + 1e-8)

    # ---- fit: one prototype per known value (mean internalized vector across contexts) ----
    def fit(self, value_vectors: Dict[str, List[torch.Tensor]]) -> None:
        rows, self.token_value, self.value_token = [], [], {}
        for name, vecs in value_vectors.items():
            if len(self.token_value) >= self.capacity:
                raise RuntimeError(f"codebook capacity {self.capacity} exceeded at value '{name}'")
            proto = self._unit(torch.stack([self._unit(v) for v in vecs], 0).mean(0))
            self.value_token[name] = len(self.token_value)
            self.token_value.append(name)
            rows.append(proto)
        self.codebook = torch.stack(rows, 0) if rows else torch.zeros(0, self.dim)

    # ---- tokenize: internalized vector -> discrete token (argmax cosine to prototypes) ----
    def tokenize(self, v: torch.Tensor) -> int:
        return int(torch.argmax(torch.matmul(self.codebook, self._unit(v))).item())

    def decode(self, token: int) -> Optional[str]:
        return self.token_value[token] if 0 <= token < len(self.token_value) else None

    # ---- same/different by TOKEN EQUALITY (exact - the discretization the arbiter needs) ----
    def same_value(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        return self.tokenize(a) == self.tokenize(b)
