# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Natural-language evaluation corpus over the organ's closed vocabulary, including
# the exact phrasings the old rule-based parser failed: F1 (novel paraphrase / fronted
# / passive FACT constructions), F3 (novel lexical-alias QUERY forms), F5 (novel query
# forms: tag / echo / indirect). Templates are the sealed v15.7a holdout families
# (steps/13_v15_7a_consolidation). Each item carries a gold (entity, attribute, value).
# F0 is a standard control. The sealed old-parser baseline on F1/F3/F5 commit_correct
# was 0.000 / 0.000 / 0.148 (paper/D_CORTEX_PAS7A_SEAL.md).

import random
from dataclasses import dataclass
from typing import Dict, List

# F1: hard FACT constructions per attribute ({V}=Capitalized value, {e}=entity, {v}=value)
F1_FACTS: Dict[str, List[str]] = {
    "color": ["{V} was the thing that defined the {e}.",
              "Among the things noticed, the {e} had a {v} tone about it.",
              "That {e}, by every account, carried {v} markings.",
              "The {e} bore {v} throughout."],
    "size": ["{V}, unmistakably, described the {e}.",
             "Comparatively, the {e} came across as {v}.",
             "In proportion, the {e} matched what one would call {v}.",
             "The {e} stood {v} beyond doubt."],
    "location": ["It was the {v} where the {e} resided.",
                 "From within the {v}, the {e} made its presence known.",
                 "Deep inside the {v}, a {e} could be seen.",
                 "The {v}, as is known, houses the {e}."],
    "state": ["{V} was what the {e} had become.",
              "Noticeably, the {e} had turned {v}.",
              "The {e}, if one paid attention, grew {v} over time.",
              "{V} described the {e} entirely."],
}
# F1: hard QUERY constructions per attribute (novel, mostly non-keyword aliases)
F1_QUERIES: Dict[str, List[str]] = {
    "color": ["As for the {e}, what attribute defined it chromatically?",
              "Regarding pigmentation of the {e}, what is it?",
              "With respect to appearance in the visible spectrum, the {e} is what?",
              "Concerning chromatic quality, the {e} is what?"],
    "size": ["In terms of proportion, the {e} is what?", "As a matter of magnitude, the {e} is what?",
             "Regarding physical scale of the {e}, what is it?", "With respect to dimension, the {e} is what?"],
    "location": ["As for whereabouts of the {e}, where is it?", "Regarding the dwelling of the {e}, where is it?",
                 "With respect to habitat, where is the {e}?", "Speaking of surroundings of the {e}, where is it?"],
    "state": ["With respect to disposition, the {e} is what?", "As a matter of temperament, the {e} is what?",
              "Regarding current disposition of the {e}, what is it?", "As for bearing of the {e}, what is it?"],
}
# F3: hard QUERY lexical-alias forms per attribute (fact stays standard)
F3_QUERIES: Dict[str, List[str]] = {
    "color": ["The {e} exhibits what pigmentation?", "What coloration characterizes the {e}?",
              "The {e} displays which wash?", "Which dye marks the {e}?"],
    "size": ["What magnitude does the {e} have?", "The {e} has what girth?",
             "Express the scale of the {e}.", "The {e} holds what proportion?"],
    "location": ["The habitat of the {e} is where?", "Identify the locale of the {e}.",
                 "The dwelling of the {e} is where?", "The quarters of the {e} are where?"],
    "state": ["What demeanor does the {e} carry?", "The bearing of the {e} is what?",
              "Describe the disposition of the {e}.", "The temperament of the {e} is what?"],
}
# F5: hard QUERY novel forms (tag / echo / indirect); fact stays standard
F5_QUERIES: Dict[str, List[str]] = {
    "color": ["The {e} is {v}, isn't it?", "The {e} is {v}? Really?",
              "I wonder what color the {e} has.", "Tell me whether the {e} is colored."],
    "size": ["The {e} is {v}, correct?", "I'd like to know the dimension of the {e}.",
             "The {e} is {v}? Surprising.", "I wonder how the {e} measures."],
    "location": ["The {e} is in the {v}, right?", "I'd like to know where the {e} resides.",
                 "The {e} is in the {v}? Interesting.", "I wonder about the habitat of the {e}."],
    "state": ["The {e} is {v}, isn't it?", "I'd like to know how the {e} is.",
              "The {e} is {v}? Truly?", "I wonder about the disposition of the {e}."],
}


@dataclass
class CorpusItem:
    family: str          # F0 / F1 / F3 / F5
    entity: str
    attribute: str
    value: str
    fact_text: str       # the sentence to ingest (write)
    query_text: str      # the question to ask
    hard_side: str       # 'fact' (F1) or 'query' (F3/F5) or 'none' (F0)


def _std_fact(e: str, a: str, v: str) -> str:
    return f"The {e} is in the {v}." if a == "location" else f"The {e} is {v}."


def _std_query(e: str, a: str) -> str:
    return f"Where is the {e}?" if a == "location" else f"What {a} is the {e}?"


def _cap(v: str) -> str:
    return v[:1].upper() + v[1:]


def _make_item(family: str, e: str, a: str, v: str, rng: random.Random) -> CorpusItem:
    if family == "F0":
        return CorpusItem("F0", e, a, v, _std_fact(e, a, v), _std_query(e, a), "none")
    if family == "F1":
        # F1 stresses BOTH sides: a fronted/passive FACT and a novel-alias QUERY
        return CorpusItem("F1", e, a, v, rng.choice(F1_FACTS[a]).format(e=e, v=v, V=_cap(v)),
                          rng.choice(F1_QUERIES[a]).format(e=e, v=v), "both")
    if family == "F3":
        return CorpusItem("F3", e, a, v, _std_fact(e, a, v), rng.choice(F3_QUERIES[a]).format(e=e, v=v), "query")
    return CorpusItem("F5", e, a, v, _std_fact(e, a, v), rng.choice(F5_QUERIES[a]).format(e=e, v=v), "query")


def build_corpus(entities: List[str], attr_values: Dict[str, List[str]],
                 n_per_family: int = 20, seed: int = 20261103) -> List[CorpusItem]:
    """Each item gets a GLOBALLY UNIQUE (entity, attribute) pair, so the gold fact set
    has no contradictory facts (no two items assert different values for the same slot).
    This keeps G_NOCORRUPT and G_RECALL free of write-conflict artifacts."""
    rng = random.Random(seed)
    attrs = ["color", "size", "location", "state"]
    pairs = [(e, a) for e in entities for a in attrs]
    rng.shuffle(pairs)
    families = ["F0", "F1", "F3", "F5"]
    need = n_per_family * len(families)
    if need > len(pairs):
        raise ValueError(f"need {need} unique (entity,attribute) pairs but only {len(pairs)} exist")
    items: List[CorpusItem] = []
    for k, (e, a) in enumerate(pairs[:need]):
        v = rng.choice(attr_values[a])
        items.append(_make_item(families[k % len(families)], e, a, v, rng))
    return items
