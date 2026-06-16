# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# IngestAdapter: replaces the old rule-based parser with a Qwen2.5-7B-Instruct
# JSON-only extractor plus a MiniLM entity resolver. Text -> FactTriple(entity,
# attribute, value, episode_id). The extractor is constrained to the organ's CLOSED
# vocabulary (attributes color/size/location/state and their value sets): any
# attribute or value outside the set is an EXTRACTION ERROR, counted as such and
# NEVER force-fit into the organ. Entities not in the known set are canonicalized by
# MiniLM cosine top-1 (raw -> canonical logged). No regex parser.

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class FactTriple:
    entity: str
    attribute: str
    value: str
    episode_id: int = 0
    raw_entity: str = ""
    resolution_cos: float = 1.0


@dataclass
class ExtractError:
    reason: str          # malformed_json / out_of_vocab_attribute / out_of_vocab_value / missing_field
    raw: str
    parsed: Optional[Dict] = None


def _extract_json(text: str) -> Optional[Dict]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        return None


class IngestAdapter:
    def __init__(self, qwen, attr_values: Dict[str, List[str]], known_entities: List[str],
                 embed_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.qwen = qwen
        self.attr_values = attr_values
        self.attributes = sorted(attr_values)
        self.known_entities = list(known_entities)
        self.log: List[Dict] = []
        from sentence_transformers import SentenceTransformer
        self._embedder = SentenceTransformer(embed_model, device="cuda")
        self._ent_emb = self._embedder.encode(self.known_entities, convert_to_tensor=True,
                                              normalize_embeddings=True)

    # ---- MiniLM entity canonicalization (with a confidence threshold) ----
    RESOLVE_THRESHOLD = 0.55     # below this the raw entity is kept (treated as unknown -> NONE_OBJECT)

    def resolve_entity(self, raw: str) -> Tuple[str, float]:
        low = raw.strip().lower()
        for e in self.known_entities:
            if e.lower() == low:
                return e, 1.0
        q = self._embedder.encode([raw], convert_to_tensor=True, normalize_embeddings=True)[0]
        sims = torch.matmul(self._ent_emb, q)
        idx = int(torch.argmax(sims).item())
        canonical, cos = self.known_entities[idx], float(sims[idx].item())
        if cos < self.RESOLVE_THRESHOLD:
            # not confidently a known entity: keep raw so the organ reports NONE_OBJECT (abstain),
            # rather than force-fitting an unknown entity onto a known slot.
            self.log.append({"event": "entity_unresolved", "raw": raw, "best": canonical, "cos": round(cos, 4)})
            return raw.strip(), cos
        self.log.append({"event": "entity_resolve", "raw": raw, "canonical": canonical, "cos": round(cos, 4)})
        return canonical, cos

    # ---- Qwen JSON extraction ----
    def _vocab_hint(self) -> str:
        return "; ".join(f"{a}: {', '.join(self.attr_values[a])}" for a in self.attributes)

    def extract_fact(self, text: str, episode_id: int = 0):
        prompt = (
            "Extract the single stated fact as JSON with EXACTLY the keys entity, attribute, value.\n"
            f"attribute MUST be EXACTLY one of: {', '.join(self.attributes)}. The sentence may use a synonym "
            "or an indirect phrasing for the attribute; map it to the closest of these four. If the stated "
            'property is genuinely none of the four, set attribute to "none".\n'
            f"value MUST be one of the allowed values for that attribute: {self._vocab_hint()}.\n"
            'Example: for "The chest shows a vivid hue." output {"entity": "chest", "attribute": "color", "value": "red"}.\n'
            "Output ONLY the JSON object with those three keys, nothing else.\n"
            f"Sentence: {text}")
        raw = self.qwen.generate_unconstrained(prompt, 60)
        obj = _extract_json(raw)
        if obj is None or "entity" not in obj or "attribute" not in obj:
            err = ExtractError("malformed_json" if obj is None else "missing_field", raw, obj)
            self.log.append({"event": "extract_fact_error", "text": text, "reason": err.reason})
            return err
        attribute = str(obj["attribute"]).strip().lower()
        if attribute == "none":
            return ExtractError("out_of_vocab_attribute", raw, obj)
        # robustness: some outputs place the value under the attribute-named key
        value_raw = obj.get("value", obj.get(attribute))
        if value_raw is None:
            self.log.append({"event": "extract_fact_error", "text": text, "reason": "missing_field"})
            return ExtractError("missing_field", raw, obj)
        value = str(value_raw).strip().lower()
        if attribute not in self.attr_values:
            return ExtractError("out_of_vocab_attribute", raw, obj)
        if value not in self.attr_values[attribute]:
            return ExtractError("out_of_vocab_value", raw, obj)
        entity, cos = self.resolve_entity(str(obj["entity"]))
        return FactTriple(entity=entity, attribute=attribute, value=value, episode_id=episode_id,
                          raw_entity=str(obj["entity"]), resolution_cos=cos)

    def extract_query(self, question: str):
        prompt = (
            "From the question, identify which attribute of which entity is being asked, as JSON with EXACTLY "
            "the keys entity, attribute.\n"
            f"attribute MUST be EXACTLY one of: {', '.join(self.attributes)}. The question may use a synonym or "
            "an indirect phrasing for the attribute; map it to the closest of these four. If the asked property "
            'is genuinely none of the four, set attribute to "none".\n'
            'Example: for "What hue does the chest show?" output {"entity": "chest", "attribute": "color"}.\n'
            "Output ONLY the JSON object with those two keys, nothing else.\n"
            f"Question: {question}")
        raw = self.qwen.generate_unconstrained(prompt, 40)
        obj = _extract_json(raw)
        if obj is None or "entity" not in obj or "attribute" not in obj:
            return ExtractError("malformed_json" if obj is None else "missing_field", raw, obj)
        attribute = str(obj["attribute"]).strip().lower()
        if attribute == "none" or attribute not in self.attr_values:
            return ExtractError("out_of_vocab_attribute", raw, obj)
        entity, cos = self.resolve_entity(str(obj["entity"]))
        return FactTriple(entity=entity, attribute=attribute, value="", raw_entity=str(obj["entity"]),
                          resolution_cos=cos)
