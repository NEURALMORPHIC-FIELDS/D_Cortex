# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Stage C - reasoning-over-memory regime. The decisive test of the vision's heart: is inference
# an OPERATION ON MEMORY (chain facts, compare values) or just token-flow? Each sequence writes
# facts to memory, then a query whose answer needs >= 2 facts. The dual-agent architecture is
# what makes this a memory test: at decode the query does NOT contain the fact text, so the
# answer MUST be read from memory. Families:
#   C1 RELATIONAL (2-hop chain): "The bear is red." + "The fox is the same color as the bear." ->
#      "What color is the fox?" -> red. The value 'red' is in NO single visible token at query
#      time (copy-proof); the model must chain fox -> bear -> red through memory.
#   C2 COMPARISON: "The bear is big." + "The fox is tiny." -> "Which is bigger, the bear or the
#      fox?" -> bear. Requires comparing two stored size values (ordered tiny<small<big<huge).
# Variants per item (the decisive gates):
#   memory        : facts in memory only (the real test).
#   text_context  : facts ALSO concatenated into the query (token-flow control - if the model is
#                   much better here, it relies on text not memory).
#   shuffled      : the relational/compared binding is permuted so the correct answer changes -
#                   genuine reasoning must FOLLOW the shuffle; a shortcut keeps the old answer.
#   unanswerable  : the chain references an entity NOT written -> the honest answer is ABSTAIN.
# This module only builds DATA (CPU, deterministic); training/eval is the cold-launch cert.

from dataclasses import dataclass
from typing import List, Optional, Tuple

ENTITIES = ["bear", "dog", "cat", "fox", "wolf", "bird", "tiger", "horse", "deer", "rabbit",
            "lion", "owl", "hawk", "mouse", "goat", "sheep", "pig", "cow", "duck", "frog"]
COLORS = ["red", "blue", "green", "yellow", "black", "white", "brown", "pink", "orange", "purple"]
SIZES = ["tiny", "small", "big", "huge"]            # ordered ascending
SIZE_RANK = {s: i for i, s in enumerate(SIZES)}
ABSTAIN = "__abstain__"


@dataclass
class CItem:
    family: str                       # C1 | C2
    variant: str                      # memory | text_context | shuffled | unanswerable
    facts: List[Tuple[str, str, str]]  # (entity, relation, value-or-target)
    fact_texts: List[str]             # surface text written to memory (in order)
    query: str                        # decode-time prompt (no fact text unless text_context)
    answer: str                       # gold answer token/value, or ABSTAIN


def _fact_text(e: str, rel: str, val: str) -> str:
    if rel == "color":
        return f"The {e} is {val}."
    if rel == "size":
        return f"The {e} is {val}."
    if rel == "same_color":
        return f"The {e} is the same color as the {val}."
    raise ValueError(rel)


def c1_relational(rng, variant: str) -> CItem:
    e1, e2 = rng.sample(ENTITIES, 2)
    v = rng.choice(COLORS)
    facts = [(e1, "color", v), (e2, "same_color", e1)]
    if variant == "shuffled":
        # a third entity gets a different color; e2 now points to it -> answer follows the shuffle
        e3 = rng.choice([x for x in ENTITIES if x not in (e1, e2)])
        v3 = rng.choice([c for c in COLORS if c != v])
        facts = [(e1, "color", v), (e3, "color", v3), (e2, "same_color", e3)]
        answer = v3
    elif variant == "unanswerable":
        facts = [(e2, "same_color", e1)]            # e1's color never written -> chain breaks
        answer = ABSTAIN
    else:
        answer = v
    texts = [_fact_text(*f) for f in facts]
    q = f"What color is the {e2}?"
    if variant == "text_context":
        q = " ".join(texts) + " " + q
    return CItem("C1", variant, facts, texts, q, answer)


def c2_comparison(rng, variant: str) -> CItem:
    e1, e2 = rng.sample(ENTITIES, 2)
    s1, s2 = rng.sample(SIZES, 2)
    facts = [(e1, "size", s1), (e2, "size", s2)]
    bigger = e1 if SIZE_RANK[s1] > SIZE_RANK[s2] else e2
    if variant == "shuffled":
        s1b, s2b = s2, s1                           # swap sizes -> bigger flips
        facts = [(e1, "size", s1b), (e2, "size", s2b)]
        bigger = e1 if SIZE_RANK[s1b] > SIZE_RANK[s2b] else e2
        answer = bigger
    elif variant == "unanswerable":
        facts = [(e1, "size", s1)]                  # e2's size never written
        answer = ABSTAIN
    else:
        answer = bigger
    texts = [_fact_text(*f) for f in facts]
    q = f"Which is bigger, the {e1} or the {e2}?"
    if variant == "text_context":
        q = " ".join(texts) + " " + q
    return CItem("C2", variant, facts, texts, q, answer)


VARIANTS = ["memory", "text_context", "shuffled", "unanswerable"]
_BUILD = {"C1": c1_relational, "C2": c2_comparison}


def build(rng, family: str, variant: str) -> CItem:
    return _BUILD[family](rng, variant)
