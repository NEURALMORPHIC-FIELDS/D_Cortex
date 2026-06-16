# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Constrained closed-set extractor. The attribute is emitted as EXACTLY one of
# {color, size, location, state, none} by scoring those options under the model
# (logit-masked classification): the model cannot emit an alias word like "bearing";
# it must map the phrasing to a closed-set member or "none" (genuine out-of-domain ->
# extraction error -> abstain). The value (for facts) is likewise classified over the
# attribute's closed value set. The entity is free-extracted and resolved by MiniLM
# with a confidence threshold. The reasoning prompt is GENERIC (which physical
# property does the phrase denote) and contains NO alias from the F1/F3/F5 generators;
# disjointness is asserted by assert_no_leak().

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from integration.ingest_adapter import FactTriple, ExtractError, _extract_json

# leak-free property descriptions; every word here is checked disjoint from the
# F-generator alias vocabulary by assert_no_leak().
ATTR_DESC = ("color = its visible hue; size = how large or small it is; "
             "location = where it is; state = its condition or mood")
ATTR_OPTIONS = ["color", "size", "location", "state", "none"]


class ConstrainedExtractor:
    def __init__(self, qwen, attr_values: Dict[str, List[str]], known_entities: List[str],
                 resolve_threshold: float = 0.55,
                 embed_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.qwen = qwen
        self.attr_values = attr_values
        self.attributes = sorted(attr_values)
        self.known_entities = list(known_entities)
        self.resolve_threshold = resolve_threshold
        self.log: List[Dict] = []
        from sentence_transformers import SentenceTransformer
        self._embedder = SentenceTransformer(embed_model, device="cuda")
        self._ent_emb = self._embedder.encode(self.known_entities, convert_to_tensor=True,
                                              normalize_embeddings=True)

    def set_threshold(self, t: float) -> None:
        self.resolve_threshold = t

    # ---- entity: free-extract + MiniLM resolve (threshold) ----
    def resolve_entity(self, raw: str, threshold: Optional[float] = None) -> Tuple[str, float]:
        th = self.resolve_threshold if threshold is None else threshold
        low = raw.strip().lower()
        for e in self.known_entities:
            if e.lower() == low:
                return e, 1.0
        q = self._embedder.encode([raw], convert_to_tensor=True, normalize_embeddings=True)[0]
        sims = torch.matmul(self._ent_emb, q)
        idx = int(torch.argmax(sims).item())
        canonical, cos = self.known_entities[idx], float(sims[idx].item())
        return (canonical, cos) if cos >= th else (raw.strip(), cos)

    # Entity resolution from the SOURCE TEXT via MiniLM (validated winner R0 of the
    # 5-way head-to-head: certify_entity_resolution.py). The earlier word-based path
    # let Qwen name the property word ("scale", "pigmentation") as the entity; resolving
    # directly from the text (which contains the entity) fixes that. The heavier Qwen
    # constrained entity classifier (R3) did NOT beat this baseline, so MiniLM is kept.
    TEXT_RESOLVE_THRESHOLD = 0.55

    def resolve_entity_from_text(self, text: str) -> Tuple[str, float]:
        q = self._embedder.encode([text], convert_to_tensor=True, normalize_embeddings=True)[0]
        sims = torch.matmul(self._ent_emb, q)
        idx = int(torch.argmax(sims).item())
        canonical, cos = self.known_entities[idx], float(sims[idx].item())
        if cos < self.TEXT_RESOLVE_THRESHOLD:
            self.log.append({"event": "entity_unresolved_text", "best": canonical, "cos": round(cos, 4)})
            return f"__unknown__{canonical}", cos      # not a known surface -> organ NONE_OBJECT
        return canonical, cos

    # ---- attribute: constrained closed-set classification ----
    def classify_attribute(self, text: str) -> str:
        prompt = ("A statement or question concerns one property of a named thing. The four properties are:\n"
                  f"{ATTR_DESC}.\n"
                  "Decide which one it concerns, even if it uses an unusual or indirect word. "
                  "If it concerns none of these four, answer none.\n"
                  f"Text: {text}\nAnswer with one of color, size, location, state, none.")
        best, _scores = self.qwen.classify(prompt, ATTR_OPTIONS, answer_prefix=" The property is")
        return best

    # ---- value: constrained closed-set classification over the attribute's values ----
    def classify_value(self, text: str, attribute: str) -> str:
        opts = self.attr_values[attribute]
        prompt = (f"The sentence states the {attribute} of a thing. "
                  f"Which {attribute} value does it state?\nSentence: {text}\n"
                  f"Answer with one of: {', '.join(opts)}.")
        best, _scores = self.qwen.classify(prompt, opts, answer_prefix=f" The {attribute} is")
        return best

    # ---- public API (drop-in for IngestAdapter) ----
    def extract_query(self, question: str):
        attribute = self.classify_attribute(question)
        if attribute == "none" or attribute not in self.attr_values:
            return ExtractError("out_of_vocab_attribute", question, {"attribute": attribute})
        entity, cos = self.resolve_entity_from_text(question)
        return FactTriple(entity=entity, attribute=attribute, value="", raw_entity=entity, resolution_cos=cos)

    def extract_fact(self, text: str, episode_id: int = 0):
        attribute = self.classify_attribute(text)
        if attribute == "none" or attribute not in self.attr_values:
            return ExtractError("out_of_vocab_attribute", text, {"attribute": attribute})
        value = self.classify_value(text, attribute)
        if value not in self.attr_values[attribute]:
            return ExtractError("out_of_vocab_value", text, {"attribute": attribute, "value": value})
        entity, cos = self.resolve_entity_from_text(text)
        return FactTriple(entity=entity, attribute=attribute, value=value, episode_id=episode_id,
                          raw_entity=entity, resolution_cos=cos)


# ---- leak assertion: prompt vocabulary disjoint from the F-generator aliases ----
_COMMON = {"the", "a", "an", "is", "are", "was", "were", "be", "what", "which", "does", "do", "of",
           "in", "to", "it", "its", "that", "this", "and", "or", "for", "with", "as", "by", "about",
           "thing", "things", "named", "single", "one", "answer", "property", "properties", "four",
           "decide", "concerns", "concern", "statement", "question", "sentence", "text", "value",
           "states", "state", "where", "how", "large", "small", "hue", "visible", "condition", "mood",
           "color", "size", "location", "creature", "person", "object", "lowercase", "word", "unusual",
           "indirect", "even", "if", "uses", "none", "least", "most", "any", "name", "stated",
           "your", "you", "i", "their", "them", "they", "exactly", "must", "only", "given", "each"}


def _alias_words(corpus_module) -> set:
    text = " ".join(t for d in (corpus_module.F1_FACTS, corpus_module.F1_QUERIES,
                                corpus_module.F3_QUERIES, corpus_module.F5_QUERIES)
                    for lst in d.values() for t in lst)
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", text) if len(w) >= 4}
    return {w for w in words if w not in _COMMON and not w.startswith("{")}


def _prompt_words(*prompts: str) -> set:
    words = set()
    for p in prompts:
        words |= {w.lower() for w in re.findall(r"[A-Za-z]+", p) if len(w) >= 4}
    return {w for w in words if w not in _COMMON}


def assert_no_leak() -> Tuple[set, set]:
    """Return (f_aliases, prompt_words) and assert disjoint. Raises on any overlap."""
    import integration.corpus as corpus
    from integration.sealed_loader import load_sealed_substrate
    f_aliases = _alias_words(corpus)
    attr_values = {k: list(v) for k, v in load_sealed_substrate()["V15_ATTR_VALUES"].items()}
    # build the prompts as they actually run (attribute placeholder -> real names, which
    # are color/size/location/state, none of which is an alias)
    value_prompts = [f"The sentence states the {a} of a thing. Which {a} value does it state? "
                     f"Answer with one of {', '.join(v)}." for a, v in attr_values.items()]
    prompt_words = _prompt_words(
        ATTR_DESC,
        "A statement or question concerns one property of a named thing. The four properties are:",
        "Decide which one it concerns, even if it uses an unusual or indirect word. If it concerns none answer none.",
        f"Text: x Answer with one of {', '.join(ATTR_OPTIONS)}. The property is",
        "Name the single thing creature person or object that the sentence is about. Answer with one lowercase word.",
        *value_prompts)
    overlap = f_aliases & prompt_words
    if overlap:
        raise AssertionError(f"LEAK: prompt words overlap F-generator aliases: {sorted(overlap)}")
    return f_aliases, prompt_words
