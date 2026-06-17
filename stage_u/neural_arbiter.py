# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Stage U - NeuralCommitArbiter: the symbolic organ's honest mechanics
# (committed / provisional / disputed + promote / retrograde / prune / reconcile),
# reimplemented to operate on CONTINUOUS value VECTORS instead of discrete value
# indices. The only thing that changes versus the symbolic Pas7a is the value-identity
# test: "are these two observations the same value?" is no longer integer equality but a
# pluggable same_value(v1, v2) over 768-dim vectors (cosine threshold, or a discretize-to-
# prototype code). Everything else - episode-counting promotion (N_promote), retrograde
# (M_retrograde), stale prune (K_prune_stale), intra-episode reconcile - is preserved.
# This module is bank-agnostic: it consumes value vectors and emits an op-sequence and a
# committed value, so it can later be wired onto DCortexV2Model's banks.

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch


def cosine_same_value(theta: float) -> Callable[[torch.Tensor, torch.Tensor], bool]:
    """Value-identity by raw cosine threshold on the continuous vectors (vision-faithful)."""
    def same(a: torch.Tensor, b: torch.Tensor) -> bool:
        an = a / (a.norm() + 1e-8)
        bn = b / (b.norm() + 1e-8)
        return float(torch.dot(an, bn)) >= theta
    return same


def prototype_same_value(prototypes: torch.Tensor) -> Callable[[torch.Tensor, torch.Tensor], bool]:
    """Value-identity by discretizing each vector to its nearest prototype, then exact-equal
    on the prototype index. Regains exact equality (a discrete code), at the cost of being
    discrete. prototypes: [K, D] unit rows."""
    P = prototypes / (prototypes.norm(dim=1, keepdim=True) + 1e-8)

    def nearest(v: torch.Tensor) -> int:
        vn = v / (v.norm() + 1e-8)
        return int(torch.argmax(torch.matmul(P, vn)).item())

    def same(a: torch.Tensor, b: torch.Tensor) -> bool:
        return nearest(a) == nearest(b)
    return same


@dataclass(eq=False)
class ValueCluster:
    """One provisional value hypothesis for a slot: a representative vector plus the SET of
    distinct episodes in which it was observed (intra-episode repeats collapse to one).
    challenger=False marks a clean FIRST write to a never-committed slot (it commits directly,
    like the symbolic CommitPath.COMMIT); challenger=True marks a value contesting an existing
    or previously-committed value (it can only take the slot via N-episode promotion)."""
    rep: torch.Tensor
    episodes: set = field(default_factory=set)
    first_ep: int = 0
    last_ep: int = 0
    challenger: bool = False


@dataclass
class Slot:
    committed: Optional[torch.Tensor] = None
    committed_episodes: set = field(default_factory=set)
    committed_first_ep: int = 0
    provisional: List[ValueCluster] = field(default_factory=list)
    ever_committed: bool = False


class NeuralCommitArbiter:
    """Continuous-value committed/provisional/disputed arbiter with Pas7a consolidation."""

    def __init__(self, same_value: Callable[[torch.Tensor, torch.Tensor], bool],
                 n_promote: int = 2, m_retrograde: int = 2,
                 k_promote_age: int = 2, k_prune_stale: int = 3) -> None:
        self.same_value = same_value
        self.N = n_promote
        self.M = m_retrograde
        self.K_age = k_promote_age
        self.K_stale = k_prune_stale
        self.slots: Dict[Tuple[str, str], Slot] = {}
        self._buffer: Dict[Tuple[str, str], List[torch.Tensor]] = {}
        self.op_counts = {"reconcile": 0, "retrograde": 0, "promote": 0, "prune": 0}

    # ---- seed a committed value directly (the L-scenarios start from a committed fact) ----
    def seed_committed(self, entity: str, attribute: str, value: torch.Tensor, episode: int) -> None:
        slot = self.slots.setdefault((entity, attribute), Slot())
        slot.committed = value.clone()
        slot.committed_episodes = {episode}
        slot.committed_first_ep = episode
        slot.ever_committed = True

    # ---- observe one (entity, attribute, value) within the current episode ----
    def observe(self, entity: str, attribute: str, value: torch.Tensor) -> None:
        self._buffer.setdefault((entity, attribute), []).append(value)

    # ---- end of episode: reconcile intra-episode, route, then consolidate ----
    def end_episode(self, episode: int) -> Dict[str, int]:
        ops = {"reconcile": 0, "retrograde": 0, "promote": 0, "prune": 0}

        # 1) RECONCILE: collapse intra-episode duplicates per slot to DISTINCT values this ep.
        for key, obs in self._buffer.items():
            distinct: List[torch.Tensor] = []
            collapsed = False
            for v in obs:
                if any(self.same_value(v, d) for d in distinct):
                    collapsed = True
                    continue
                distinct.append(v)
            if collapsed:
                ops["reconcile"] += 1
            slot = self.slots.setdefault(key, Slot())
            # 2) ROUTE each distinct value: reinforce committed, or add/extend a provisional
            # cluster. The challenger flag records whether the slot was committed/ever-committed
            # when the value first appeared (a contest) or empty-and-virgin (a clean first write).
            for v in distinct:
                if slot.committed is not None and self.same_value(v, slot.committed):
                    slot.committed_episodes.add(episode)
                    continue
                matched = None
                for c in slot.provisional:
                    if self.same_value(v, c.rep):
                        matched = c
                        break
                if matched is None:
                    is_challenger = slot.committed is not None or slot.ever_committed
                    slot.provisional.append(ValueCluster(rep=v.clone(), episodes={episode},
                                                         first_ep=episode, last_ep=episode,
                                                         challenger=is_challenger))
                else:
                    matched.episodes.add(episode)
                    matched.last_ep = episode
        self._buffer.clear()

        # 3) consolidate every slot
        for slot in self.slots.values():
            # CLEAN COMMIT: a single clean (non-challenger) first write to a virgin empty slot
            # commits directly (the symbolic CommitPath.COMMIT). Not counted as a consolidation op.
            if slot.committed is None and not slot.ever_committed:
                clean = [c for c in slot.provisional if not c.challenger]
                if len(clean) == 1 and len(slot.provisional) == 1:
                    c = clean[0]
                    slot.committed = c.rep.clone()
                    slot.committed_episodes = set(c.episodes)
                    slot.committed_first_ep = c.first_ep
                    slot.ever_committed = True
                    slot.provisional.remove(c)
            # RETROGRADE: a challenger confirmed in >= M distinct episodes demotes committed
            if slot.committed is not None:
                for c in slot.provisional:
                    if len(c.episodes) >= self.M:
                        slot.committed = None
                        slot.committed_episodes = set()
                        ops["retrograde"] += 1
                        break
            # PROMOTE: a provisional with >= N distinct episodes AND age >= K_age takes the slot
            if slot.committed is None:
                promoted = None
                for c in slot.provisional:
                    if len(c.episodes) >= self.N and (episode - c.first_ep) >= self.K_age:
                        promoted = c
                        break
                if promoted is not None:
                    slot.committed = promoted.rep.clone()
                    slot.committed_episodes = set(promoted.episodes)
                    slot.committed_first_ep = promoted.first_ep
                    slot.ever_committed = True
                    slot.provisional.remove(promoted)
                    ops["promote"] += 1
            # PRUNE: provisional silent for >= K_stale episodes is dropped
            survivors = []
            for c in slot.provisional:
                if (episode - c.last_ep) >= self.K_stale:
                    ops["prune"] += 1
                else:
                    survivors.append(c)
            slot.provisional = survivors

        for k in ops:
            self.op_counts[k] += ops[k]
        return ops

    # ---- read the committed value (or None) ----
    def read(self, entity: str, attribute: str) -> Optional[torch.Tensor]:
        slot = self.slots.get((entity, attribute))
        return None if slot is None else slot.committed
