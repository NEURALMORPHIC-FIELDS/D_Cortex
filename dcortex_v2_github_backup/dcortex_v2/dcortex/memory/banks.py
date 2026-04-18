# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha
# Memory Banks: State, EpisodeObject, Conflict, Archive, Working, EpisodeSSM.
# Patent EP25216372.0.

from typing import Optional

import torch
import torch.nn as nn


# ======================================================================
# BASE BANK
# ======================================================================

class MemoryBank(nn.Module):
    """Slot-based memory with differentiable overlay for gradient flow.

    Keys and values are stored as buffers (persistent, no grad).
    During a training episode, the writer additionally stores grad-carrying
    tensors in an overlay dict. Readers use get_diff_*() methods to get
    tensors that combine buffer (no grad) and overlay (with grad).
    After backward(), clear_overlay() detaches everything.

    This enables gradient from decoder loss to flow through memory values
    back to the encoder's writer heads, without requiring persistent
    computation graphs across episodes.
    """

    def __init__(
        self,
        capacity: int,
        hidden_dim: int,
        d_ent: int,
        d_rel: int,
        d_typ: int,
    ) -> None:
        super().__init__()
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.d_ent = d_ent
        self.d_rel = d_rel
        self.d_typ = d_typ

        self.register_buffer("k_ent", torch.zeros(capacity, d_ent))
        self.register_buffer("k_rel", torch.zeros(capacity, d_rel))
        self.register_buffer("k_typ", torch.zeros(capacity, d_typ))
        self.register_buffer("values", torch.zeros(capacity, hidden_dim))
        self.register_buffer("occupied", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("usage", torch.zeros(capacity))
        self.register_buffer(
            "last_write_step",
            torch.full((capacity,), -1, dtype=torch.long),
        )

        # Differentiable overlay: {slot_idx: {value, k_ent, k_rel, k_typ}}
        # Populated by writer WITH grad, used by reader for gradient flow.
        self._overlay: dict = {}

    def set_overlay(
        self,
        idx: int,
        value: torch.Tensor,
        k_ent: torch.Tensor,
        k_rel: torch.Tensor,
        k_typ: torch.Tensor,
    ) -> None:
        """Store grad-carrying tensors for current episode."""
        self._overlay[idx] = {
            'value': value, 'k_ent': k_ent, 'k_rel': k_rel, 'k_typ': k_typ,
        }

    def clear_overlay(self) -> None:
        """Remove all overlay entries. Call after backward()."""
        self._overlay.clear()

    def get_diff_values(self) -> torch.Tensor:
        """Return [C, D] values with overlay rows carrying grad."""
        if not self._overlay:
            return self.values
        rows = []
        for i in range(self.capacity):
            if i in self._overlay:
                rows.append(self._overlay[i]['value'])
            else:
                rows.append(self.values[i])
        return torch.stack(rows)

    def get_diff_k_ent(self) -> torch.Tensor:
        if not self._overlay:
            return self.k_ent
        rows = []
        for i in range(self.capacity):
            if i in self._overlay:
                rows.append(self._overlay[i]['k_ent'])
            else:
                rows.append(self.k_ent[i])
        return torch.stack(rows)

    def get_diff_k_rel(self) -> torch.Tensor:
        if not self._overlay:
            return self.k_rel
        rows = []
        for i in range(self.capacity):
            if i in self._overlay:
                rows.append(self._overlay[i]['k_rel'])
            else:
                rows.append(self.k_rel[i])
        return torch.stack(rows)

    def get_diff_k_typ(self) -> torch.Tensor:
        if not self._overlay:
            return self.k_typ
        rows = []
        for i in range(self.capacity):
            if i in self._overlay:
                rows.append(self._overlay[i]['k_typ'])
            else:
                rows.append(self.k_typ[i])
        return torch.stack(rows)

    def reset(self) -> None:
        """Clear all slots and overlay."""
        self.k_ent.zero_()
        self.k_rel.zero_()
        self.k_typ.zero_()
        self.values.zero_()
        self.occupied.zero_()
        self.usage.zero_()
        self.last_write_step.fill_(-1)
        self._overlay.clear()

    def n_occupied(self) -> int:
        return int(self.occupied.sum().item())

    def free_slot(self) -> int:
        """Return index of first free slot, or -1 if full."""
        free = (~self.occupied).nonzero(as_tuple=False)
        if free.numel() == 0:
            return -1
        return int(free[0].item())

    def lru_slot(self) -> int:
        """Return least-recently-used OCCUPIED slot.

        Falls back to slot 0 if no slots are occupied (defensive).
        """
        if self.n_occupied() == 0:
            return 0
        steps = self.last_write_step.float().clone()
        steps[~self.occupied] = float("inf")
        return int(steps.argmin().item())

    def snapshot(self) -> dict:
        """Diagnostic dictionary. Not used in forward."""
        return {
            "capacity": self.capacity,
            "occupied": self.n_occupied(),
            "usage_mean": float(self.usage[self.occupied].mean().item())
            if self.n_occupied() > 0 else 0.0,
            "usage_max": float(self.usage.max().item()),
        }


# ======================================================================
# CONCRETE BANKS
# ======================================================================

class StateMemory(MemoryBank):
    """Slot-based factual / stable memory.

    Holds the model's view of stable facts and ground-truth-like state.
    Populated by the writer through gating over the hidden stream.
    Consolidated (promoted) to ArchiveMemory on decay.
    """


class EpisodeObjectMemory(MemoryBank):
    """Discrete episodic objects.

    Holds events, scenes, or individuated context objects produced during
    a conversation. Read alongside EpisodeSSM by the EpisodeReader.
    """


class ConflictMemory(MemoryBank):
    """Difference-vector memory for contradictions.

    When a write candidate has high key-similarity but large value
    divergence with an existing state slot, the difference
    (candidate_value - existing_value) is written here rather than
    overwriting state. Preserves both facts for downstream resolution.
    """


class ArchiveMemory(MemoryBank):
    """Long-term consolidated storage.

    Target for slots migrated out of StateMemory by the consolidator.
    Larger capacity, lower write frequency.
    """


class WorkingMemory(MemoryBank):
    """Rolling short-term memory for the current turn or conversation.

    Small capacity, aggressively overwritten via LRU. Provides live
    recent-context recall inside the current session.
    """


# ======================================================================
# EPISODE SSM (trainable state-space recurrence)
# ======================================================================

class EpisodeSSM(nn.Module):
    """Continuous episodic state as a trainable state-space recurrence.

    Recurrence:
        x_t = sigmoid(a) * x_{t-1} + B * phi(u_t)

    Readout:
        r_ssm = C * x_t

    Parameters a, B, C are learned. phi is a GELU nonlinearity.
    The state `x` is a persistent buffer: within a forward pass gradients
    flow through a, B, C; across forward passes the state is detached
    so no graph persists across conversations or turns.
    """

    def __init__(self, input_dim: int, state_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim

        # Recurrent gate, per-dim scalar through sigmoid
        self.a_raw = nn.Parameter(torch.zeros(state_dim))

        # Input projection B with phi=GELU
        self.B = nn.Linear(input_dim, state_dim)
        self.phi = nn.GELU()

        # Output projection C
        self.C = nn.Linear(state_dim, input_dim)

        # Persistent state, session-scoped
        self.register_buffer("x", torch.zeros(state_dim))

        # Readout buffer: updated after each forward(), readable by decoder
        # without gradient flow through encoder parameters.
        self.register_buffer("readout", torch.zeros(input_dim))

    def reset(self) -> None:
        self.x.zero_()
        self.readout.zero_()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Advance SSM one step and return the current readout.

        Args:
            u: representative input vector, shape [input_dim] or
               [B, input_dim]. If batched, inputs are averaged across
               the batch (single shared SSM state).

        Returns:
            Readout r_ssm, shape [input_dim].
        """
        if u.dim() == 2:
            u = u.mean(dim=0)
        elif u.dim() != 1:
            raise ValueError(f"EpisodeSSM input must be 1D or 2D, got {u.dim()}D")

        a = torch.sigmoid(self.a_raw)                # [state_dim]
        drive = self.B(self.phi(u))                  # [state_dim]
        x_new = a * self.x.detach() + drive          # [state_dim]
        self.x.data = x_new.detach()
        r = self.C(x_new)                            # [input_dim]
        # Store readout as buffer for decoder (no grad)
        self.readout.data = r.detach()
        return r                                     # [input_dim]

    def get_readout(self) -> torch.Tensor:
        """Return the last readout as a detached buffer.

        Used by the decoder to read SSM state without gradient flow
        through encoder parameters. Returns [input_dim].
        """
        return self.readout.detach()
