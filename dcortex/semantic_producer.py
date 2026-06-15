# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b
# Conservative latent-prototype semantic hypothesis producer.

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from dcortex.semantic_adapter import (
    AdapterDecision,
    ConservativeSemanticAdapter,
    HypothesisMode,
    RequestedDestination,
    SemanticHypothesis,
)


class EmbeddingBackend(ABC):
    """Abstract semantic embedding backend."""

    backend_id: str
    backend_version: str

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        """Return a finite `[N, D]` float tensor for the supplied texts."""


class CandidateScoringBackend(ABC):
    """Abstract conditional-likelihood scoring backend."""

    backend_id: str
    backend_version: str

    @abstractmethod
    def score(
        self,
        source_text: str,
        prompt_template: str,
        candidates: Sequence[str],
    ) -> torch.Tensor:
        """Return one finite conditional score per canonical candidate."""


class SemanticClassificationBackend(ABC):
    """Abstract trained semantic classification backend."""

    backend_id: str
    backend_version: str
    entity_ids: Tuple[str, ...]
    attribute_ids: Tuple[str, ...]
    unknown_entity_id: str
    unknown_attribute_id: str

    @abstractmethod
    def classify(self, texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return entity and attribute probability tensors for each text."""


class SemanticFeatureBackend(ABC):
    """Abstract frozen semantic feature backend."""

    backend_id: str
    backend_version: str

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the feature width."""

    @abstractmethod
    def features(self, texts: Sequence[str]) -> torch.Tensor:
        """Return one finite feature vector per text."""


class DCortexTokenEmbeddingBackend(EmbeddingBackend):
    """Mean-pooled D_Cortex shared-token embedding backend."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Callable[[str], List[int]],
        max_seq_len: int = 64,
        backend_version: str = "1.0",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.backend_id = "dcortex_shared_token_mean"
        self.backend_version = backend_version

    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        """Embed text by masked mean pooling over shared token embeddings."""
        if not texts:
            raise ValueError("texts must not be empty")
        device = next(self.model.parameters()).device
        encoded = [self.tokenizer(text)[: self.max_seq_len] for text in texts]
        if any(not item for item in encoded):
            raise ValueError("every text must produce at least one token")
        width = max(len(item) for item in encoded)
        ids = torch.zeros(len(encoded), width, dtype=torch.long, device=device)
        mask = torch.zeros(len(encoded), width, dtype=torch.float32, device=device)
        for row, tokens in enumerate(encoded):
            ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
            mask[row, : len(tokens)] = 1.0
        with torch.no_grad():
            embeddings = self.model.shared_token_emb(ids).float()
            pooled = (embeddings * mask.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return pooled


class DCortexCausalLikelihoodBackend(CandidateScoringBackend):
    """Read-only D_Cortex decoder conditional-likelihood backend."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Callable[[str], List[int]],
        pad_token_id: int,
        max_seq_len: int = 128,
        backend_version: str = "1.0",
    ) -> None:
        if max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2")
        self.model = model
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.backend_id = "dcortex_decoder_causal_likelihood"
        self.backend_version = backend_version

    def score(
        self,
        source_text: str,
        prompt_template: str,
        candidates: Sequence[str],
    ) -> torch.Tensor:
        """Score canonical continuations without writing model memory."""
        if not source_text.strip():
            raise ValueError("source_text must not be empty")
        if "{source_text}" not in prompt_template:
            raise ValueError("prompt_template must contain {source_text}")
        if not candidates or any(not candidate.strip() for candidate in candidates):
            raise ValueError("candidates must contain non-empty strings")

        prompt_ids = self.tokenizer(prompt_template.format(source_text=source_text))
        if not prompt_ids:
            raise ValueError("prompt_template must produce at least one token")

        sequences: List[List[int]] = []
        candidate_starts: List[int] = []
        candidate_ids: List[List[int]] = []
        for candidate in candidates:
            continuation = self.tokenizer(" " + candidate.strip())
            if not continuation:
                raise ValueError("every candidate must produce at least one token")
            max_prompt = self.max_seq_len - len(continuation)
            if max_prompt < 1:
                raise ValueError("candidate exceeds max_seq_len")
            retained_prompt = prompt_ids[-max_prompt:]
            candidate_starts.append(len(retained_prompt))
            candidate_ids.append(continuation)
            sequences.append(retained_prompt + continuation)

        device = next(self.model.parameters()).device
        width = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), width),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(sequences), width), dtype=torch.long, device=device
        )
        for row, sequence in enumerate(sequences):
            input_ids[row, : len(sequence)] = torch.tensor(
                sequence, dtype=torch.long, device=device
            )
            attention_mask[row, : len(sequence)] = 1

        with torch.inference_mode():
            logits = self.model.decode(input_ids, attention_mask=attention_mask)
            log_probs = F.log_softmax(logits.float(), dim=-1)
        scores: List[torch.Tensor] = []
        for row, continuation in enumerate(candidate_ids):
            start = candidate_starts[row]
            token_scores = [
                log_probs[row, start + offset - 1, token_id]
                for offset, token_id in enumerate(continuation)
            ]
            scores.append(torch.stack(token_scores).mean())
        return torch.stack(scores).detach().cpu()


class DCortexPooledFeatureBackend(SemanticFeatureBackend):
    """Frozen five-view pooling over D_Cortex shared-token embeddings."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Callable[[str], List[int]],
        max_seq_len: int = 128,
        batch_size: int = 256,
        backend_version: str = "1.0",
    ) -> None:
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.backend_id = "dcortex_frozen_token_five_view_pool"
        self.backend_version = backend_version

    @property
    def output_dim(self) -> int:
        """Return the concatenated pooled-feature width."""
        return int(self.model.shared_token_emb.embedding_dim) * 5

    def features(self, texts: Sequence[str]) -> torch.Tensor:
        """Return mean, max, min, first, and last frozen embedding views."""
        if not texts:
            raise ValueError("texts must not be empty")
        device = next(self.model.parameters()).device
        batches: List[torch.Tensor] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            encoded = [self.tokenizer(text)[: self.max_seq_len] for text in batch_texts]
            if any(not item for item in encoded):
                raise ValueError("every text must produce at least one token")
            width = max(len(item) for item in encoded)
            ids = torch.zeros(len(encoded), width, dtype=torch.long, device=device)
            mask = torch.zeros(len(encoded), width, dtype=torch.bool, device=device)
            for row, tokens in enumerate(encoded):
                ids[row, : len(tokens)] = torch.tensor(
                    tokens, dtype=torch.long, device=device
                )
                mask[row, : len(tokens)] = True
            with torch.no_grad():
                embeddings = self.model.shared_token_emb(ids).float()
                mask_float = mask.float()
                mean = (embeddings * mask_float.unsqueeze(-1)).sum(dim=1)
                mean = mean / mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
                maximum = embeddings.masked_fill(
                    ~mask.unsqueeze(-1), float("-inf")
                ).max(dim=1).values
                minimum = embeddings.masked_fill(
                    ~mask.unsqueeze(-1), float("inf")
                ).min(dim=1).values
                first = embeddings[:, 0]
                last = embeddings[
                    torch.arange(len(encoded), device=device),
                    mask.sum(dim=1) - 1,
                ]
                batches.append(torch.cat((mean, maximum, minimum, first, last), dim=1))
        return torch.cat(batches, dim=0)


class DCortexContextualFeatureBackend(SemanticFeatureBackend):
    """Frozen order-sensitive features from D_Cortex decoder standard blocks."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Callable[[str], List[int]],
        max_seq_len: int = 128,
        batch_size: int = 128,
        backend_version: str = "1.0",
    ) -> None:
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        required = (
            "shared_token_emb",
            "shared_pos_emb",
            "dec_emb_norm",
            "dec_emb_drop",
            "dec_standard_blocks",
        )
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise ValueError(f"contextual model lacks required attributes: {missing}")
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.backend_id = "dcortex_frozen_contextual_five_view_pool"
        self.backend_version = backend_version

    @property
    def output_dim(self) -> int:
        """Return the concatenated contextual feature width."""
        return int(self.model.shared_token_emb.embedding_dim) * 5

    def features(self, texts: Sequence[str]) -> torch.Tensor:
        """Return five pooled views after frozen order-sensitive processing."""
        if not texts:
            raise ValueError("texts must not be empty")
        device = next(self.model.parameters()).device
        batches: List[torch.Tensor] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            encoded = [self.tokenizer(text)[: self.max_seq_len] for text in batch_texts]
            if any(not item for item in encoded):
                raise ValueError("every text must produce at least one token")
            width = max(len(item) for item in encoded)
            ids = torch.zeros(len(encoded), width, dtype=torch.long, device=device)
            mask = torch.zeros(len(encoded), width, dtype=torch.bool, device=device)
            for row, tokens in enumerate(encoded):
                ids[row, : len(tokens)] = torch.tensor(
                    tokens, dtype=torch.long, device=device
                )
                mask[row, : len(tokens)] = True
            positions = torch.arange(width, device=device).unsqueeze(0)
            with torch.inference_mode():
                hidden = self.model.shared_token_emb(ids) + self.model.shared_pos_emb(
                    positions
                )
                hidden = self.model.dec_emb_drop(self.model.dec_emb_norm(hidden))
                for block in self.model.dec_standard_blocks:
                    hidden = block(hidden, mask.long())
                hidden = hidden.float()
                mask_float = mask.float()
                mean = (hidden * mask_float.unsqueeze(-1)).sum(dim=1)
                mean = mean / mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
                maximum = hidden.masked_fill(
                    ~mask.unsqueeze(-1), float("-inf")
                ).max(dim=1).values
                minimum = hidden.masked_fill(
                    ~mask.unsqueeze(-1), float("inf")
                ).min(dim=1).values
                first = hidden[:, 0]
                last = hidden[
                    torch.arange(len(encoded), device=device),
                    mask.sum(dim=1) - 1,
                ]
                batches.append(torch.cat((mean, maximum, minimum, first, last), dim=1))
        return torch.cat(batches, dim=0)


class SemanticQueryHead(torch.nn.Module):
    """Separate trainable entity and attribute MLPs over frozen pooled features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        entity_classes: int,
        attribute_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, entity_classes, attribute_classes) < 1:
            raise ValueError("all dimensions and class counts must be positive")
        self.entity_network = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, entity_classes),
        )
        self.attribute_network = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, attribute_classes),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return entity and attribute logits."""
        return self.entity_network(features), self.attribute_network(features)


class PooledSemanticClassificationBackend(SemanticClassificationBackend):
    """Trained classifier over frozen D_Cortex pooled token features."""

    def __init__(
        self,
        feature_backend: SemanticFeatureBackend,
        head: SemanticQueryHead,
        entity_ids: Sequence[str],
        attribute_ids: Sequence[str],
        unknown_entity_id: str,
        unknown_attribute_id: str,
        backend_version: str = "1.0",
    ) -> None:
        if len(entity_ids) != head.entity_network[-1].out_features:
            raise ValueError("entity_ids count must match the entity head")
        if len(attribute_ids) != head.attribute_network[-1].out_features:
            raise ValueError("attribute_ids count must match the attribute head")
        if unknown_entity_id not in entity_ids:
            raise ValueError("unknown_entity_id must be present in entity_ids")
        if unknown_attribute_id not in attribute_ids:
            raise ValueError("unknown_attribute_id must be present in attribute_ids")
        self.feature_backend = feature_backend
        self.head = head
        self.entity_ids = tuple(entity_ids)
        self.attribute_ids = tuple(attribute_ids)
        self.unknown_entity_id = unknown_entity_id
        self.unknown_attribute_id = unknown_attribute_id
        self.backend_id = "dcortex_trained_pooled_semantic_classifier"
        self.backend_version = backend_version

    def classify(self, texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return deterministic entity and attribute probabilities."""
        features = self.feature_backend.features(texts)
        device = next(self.head.parameters()).device
        with torch.inference_mode():
            entity_logits, attribute_logits = self.head(features.to(device))
            entity_probabilities = torch.softmax(entity_logits.float(), dim=-1)
            attribute_probabilities = torch.softmax(attribute_logits.float(), dim=-1)
        return entity_probabilities.cpu(), attribute_probabilities.cpu()


@dataclass(frozen=True)
class AxisMatch:
    """Top semantic match for one interpretation axis."""

    axis: str
    selected_id: Optional[str]
    top_score: float
    second_score: float
    margin: float
    threshold: float
    margin_threshold: float
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the axis match."""
        return asdict(self)


@dataclass(frozen=True)
class ProducerResult:
    """Conservative producer result, including abstentions."""

    emitted: bool
    reason_codes: Tuple[str, ...]
    matches: Tuple[AxisMatch, ...]
    hypothesis: Optional[SemanticHypothesis]
    adapter_decision: Optional[AdapterDecision]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the producer result."""
        return {
            "emitted": self.emitted,
            "reason_codes": list(self.reason_codes),
            "matches": [match.to_dict() for match in self.matches],
            "hypothesis": None if self.hypothesis is None else self.hypothesis.to_dict(),
            "adapter_decision": (
                None if self.adapter_decision is None else self.adapter_decision.to_dict()
            ),
        }


@dataclass(frozen=True)
class LikelihoodAxisScore:
    """Ranked canonical candidates for one likelihood interpretation axis."""

    axis: str
    selected_id: Optional[str]
    top_probability: float
    second_probability: float
    margin: float
    margin_threshold: float
    minimum_probability: float
    passed: bool
    candidate_probabilities: Tuple[Tuple[str, float], ...]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the likelihood axis score."""
        data = asdict(self)
        data["candidate_probabilities"] = {
            candidate_id: probability
            for candidate_id, probability in self.candidate_probabilities
        }
        return data


@dataclass(frozen=True)
class LikelihoodProducerResult:
    """Conservative likelihood-producer result, including abstentions."""

    emitted: bool
    reason_codes: Tuple[str, ...]
    scores: Tuple[LikelihoodAxisScore, ...]
    hypothesis: Optional[SemanticHypothesis]
    adapter_decision: Optional[AdapterDecision]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the likelihood-producer result."""
        return {
            "emitted": self.emitted,
            "reason_codes": list(self.reason_codes),
            "scores": [score.to_dict() for score in self.scores],
            "hypothesis": None if self.hypothesis is None else self.hypothesis.to_dict(),
            "adapter_decision": (
                None if self.adapter_decision is None else self.adapter_decision.to_dict()
            ),
        }


@dataclass(frozen=True)
class MultiViewAxisScore:
    """Aggregated likelihood evidence from multiple independent prompt views."""

    axis: str
    selected_id: Optional[str]
    top_probability: float
    second_probability: float
    margin: float
    margin_threshold: float
    consensus_count: int
    minimum_consensus: int
    total_views: int
    passed: bool
    candidate_probabilities: Tuple[Tuple[str, float], ...]
    view_votes: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the multi-view axis score."""
        data = asdict(self)
        data["candidate_probabilities"] = {
            candidate_id: probability
            for candidate_id, probability in self.candidate_probabilities
        }
        data["view_votes"] = list(self.view_votes)
        return data


@dataclass(frozen=True)
class MultiViewProducerResult:
    """Conservative multi-view likelihood result, including abstentions."""

    emitted: bool
    reason_codes: Tuple[str, ...]
    scores: Tuple[MultiViewAxisScore, ...]
    hypothesis: Optional[SemanticHypothesis]
    adapter_decision: Optional[AdapterDecision]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the multi-view producer result."""
        return {
            "emitted": self.emitted,
            "reason_codes": list(self.reason_codes),
            "scores": [score.to_dict() for score in self.scores],
            "hypothesis": None if self.hypothesis is None else self.hypothesis.to_dict(),
            "adapter_decision": (
                None if self.adapter_decision is None else self.adapter_decision.to_dict()
            ),
        }


@dataclass(frozen=True)
class ClassificationAxisScore:
    """Ranked trained-classifier probabilities for one semantic axis."""

    axis: str
    selected_id: str
    top_probability: float
    second_probability: float
    margin: float
    margin_threshold: float
    unknown_selected: bool
    passed: bool
    candidate_probabilities: Tuple[Tuple[str, float], ...]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the classification axis score."""
        data = asdict(self)
        data["candidate_probabilities"] = {
            candidate_id: probability
            for candidate_id, probability in self.candidate_probabilities
        }
        return data


@dataclass(frozen=True)
class ClassificationProducerResult:
    """Conservative trained-classifier result, including abstentions."""

    emitted: bool
    reason_codes: Tuple[str, ...]
    scores: Tuple[ClassificationAxisScore, ...]
    hypothesis: Optional[SemanticHypothesis]
    adapter_decision: Optional[AdapterDecision]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the trained-classifier producer result."""
        return {
            "emitted": self.emitted,
            "reason_codes": list(self.reason_codes),
            "scores": [score.to_dict() for score in self.scores],
            "hypothesis": None if self.hypothesis is None else self.hypothesis.to_dict(),
            "adapter_decision": (
                None if self.adapter_decision is None else self.adapter_decision.to_dict()
            ),
        }


class ConservativePrototypeProducer:
    """Produce hypotheses only when every required semantic axis is unambiguous."""

    REASON_EMITTED = "EMITTED"
    REASON_EMPTY_CANDIDATES = "EMPTY_CANDIDATES"
    REASON_BELOW_THRESHOLD = "BELOW_THRESHOLD"
    REASON_MARGIN_TOO_SMALL = "MARGIN_TOO_SMALL"
    REASON_NONFINITE_EMBEDDING = "NONFINITE_EMBEDDING"

    def __init__(
        self,
        backend: EmbeddingBackend,
        adapter: ConservativeSemanticAdapter,
        similarity_threshold: float = 0.55,
        margin_threshold: float = 0.05,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")
        if not 0.0 <= margin_threshold <= 1.0:
            raise ValueError("margin_threshold must be in [0, 1]")
        self.backend = backend
        self.adapter = adapter
        self.similarity_threshold = similarity_threshold
        self.margin_threshold = margin_threshold

    def _match_axis(
        self,
        axis: str,
        source_vector: torch.Tensor,
        prototypes: Mapping[str, Sequence[str]],
    ) -> AxisMatch:
        if not prototypes or any(not texts for texts in prototypes.values()):
            return AxisMatch(
                axis=axis,
                selected_id=None,
                top_score=0.0,
                second_score=0.0,
                margin=0.0,
                threshold=self.similarity_threshold,
                margin_threshold=self.margin_threshold,
                passed=False,
            )
        concept_ids = sorted(prototypes)
        flat_texts: List[str] = []
        spans: List[Tuple[int, int]] = []
        for concept_id in concept_ids:
            start = len(flat_texts)
            flat_texts.extend(prototypes[concept_id])
            spans.append((start, len(flat_texts)))
        vectors = self.backend.embed(flat_texts).float()
        if not torch.isfinite(vectors).all() or not torch.isfinite(source_vector).all():
            return AxisMatch(
                axis=axis,
                selected_id=None,
                top_score=0.0,
                second_score=0.0,
                margin=0.0,
                threshold=self.similarity_threshold,
                margin_threshold=self.margin_threshold,
                passed=False,
            )
        vectors = F.normalize(vectors, dim=-1)
        source = F.normalize(source_vector.float().unsqueeze(0), dim=-1)[0]
        centroids = []
        for start, end in spans:
            centroids.append(F.normalize(vectors[start:end].mean(dim=0), dim=-1))
        centroid_tensor = torch.stack(centroids)
        cosine = centroid_tensor @ source
        scores = ((cosine + 1.0) / 2.0).detach().cpu().tolist()
        ranked = sorted(
            zip(concept_ids, scores), key=lambda item: (-item[1], item[0])
        )
        selected_id, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_score - second_score
        passed = (
            top_score >= self.similarity_threshold and margin >= self.margin_threshold
        )
        return AxisMatch(
            axis=axis,
            selected_id=selected_id,
            top_score=round(float(top_score), 6),
            second_score=round(float(second_score), 6),
            margin=round(float(margin), 6),
            threshold=self.similarity_threshold,
            margin_threshold=self.margin_threshold,
            passed=passed,
        )

    def produce(
        self,
        hypothesis_id: str,
        episode_id: int,
        mode: HypothesisMode,
        source_text: str,
        entity_prototypes: Mapping[str, Sequence[str]],
        attr_prototypes: Mapping[str, Sequence[str]],
        value_prototypes: Optional[Mapping[str, Sequence[str]]] = None,
        provenance: Tuple[str, ...] = (),
    ) -> ProducerResult:
        """Produce and submit one conservative semantic hypothesis."""
        source_vector = self.backend.embed([source_text]).float()[0]
        axes = [
            self._match_axis("entity", source_vector, entity_prototypes),
            self._match_axis("attribute", source_vector, attr_prototypes),
        ]
        if mode == HypothesisMode.FACT:
            axes.append(
                self._match_axis("value", source_vector, value_prototypes or {})
            )

        reasons: List[str] = []
        for match in axes:
            if match.selected_id is None:
                reasons.append(f"{self.REASON_EMPTY_CANDIDATES}:{match.axis}")
            elif match.top_score < match.threshold:
                reasons.append(f"{self.REASON_BELOW_THRESHOLD}:{match.axis}")
            elif match.margin < match.margin_threshold:
                reasons.append(f"{self.REASON_MARGIN_TOO_SMALL}:{match.axis}")
        if not torch.isfinite(source_vector).all():
            reasons.append(self.REASON_NONFINITE_EMBEDDING)
        if reasons:
            return ProducerResult(
                emitted=False,
                reason_codes=tuple(sorted(set(reasons))),
                matches=tuple(axes),
                hypothesis=None,
                adapter_decision=None,
            )

        selected = {match.axis: str(match.selected_id) for match in axes}
        confidence = min(match.top_score for match in axes)
        uncertainty = 1.0 - confidence
        producer_provenance = tuple(provenance) + (
            f"backend:{self.backend.backend_id}:{self.backend.backend_version}",
        )
        destination = (
            RequestedDestination.PROVISIONAL_ONLY
            if mode == HypothesisMode.FACT
            else RequestedDestination.QUERY_ONLY
        )
        hypothesis = SemanticHypothesis(
            hypothesis_id=hypothesis_id,
            episode_id=episode_id,
            mode=mode,
            source_text=source_text,
            producer="conservative_prototype_producer",
            producer_version="1.0",
            provenance=producer_provenance,
            confidence=confidence,
            uncertainty=uncertainty,
            requested_destination=destination,
            entity_id=selected["entity"],
            attr_type=selected["attribute"],
            value_id=selected.get("value"),
        )
        decision = self.adapter.submit(hypothesis)
        return ProducerResult(
            emitted=True,
            reason_codes=(self.REASON_EMITTED,),
            matches=tuple(axes),
            hypothesis=hypothesis,
            adapter_decision=decision,
        )


class ConservativeLikelihoodQueryProducer:
    """Produce read-only query interpretations from canonical likelihood ranks."""

    REASON_EMITTED = "EMITTED"
    REASON_EMPTY_CANDIDATES = "EMPTY_CANDIDATES"
    REASON_BELOW_PROBABILITY = "BELOW_PROBABILITY"
    REASON_MARGIN_TOO_SMALL = "MARGIN_TOO_SMALL"
    REASON_NONFINITE_SCORE = "NONFINITE_SCORE"

    def __init__(
        self,
        backend: CandidateScoringBackend,
        adapter: ConservativeSemanticAdapter,
        entity_prompt: str,
        attribute_prompt: str,
        entity_margin_threshold: float,
        attribute_margin_threshold: float,
        minimum_probability: float = 0.0,
    ) -> None:
        for name, threshold in (
            ("entity_margin_threshold", entity_margin_threshold),
            ("attribute_margin_threshold", attribute_margin_threshold),
            ("minimum_probability", minimum_probability),
        ):
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if "{source_text}" not in entity_prompt or "{source_text}" not in attribute_prompt:
            raise ValueError("prompts must contain {source_text}")
        self.backend = backend
        self.adapter = adapter
        self.entity_prompt = entity_prompt
        self.attribute_prompt = attribute_prompt
        self.entity_margin_threshold = entity_margin_threshold
        self.attribute_margin_threshold = attribute_margin_threshold
        self.minimum_probability = minimum_probability

    def _score_axis(
        self,
        axis: str,
        source_text: str,
        prompt: str,
        candidates: Mapping[str, str],
        margin_threshold: float,
    ) -> LikelihoodAxisScore:
        if not candidates or any(not value.strip() for value in candidates.values()):
            return LikelihoodAxisScore(
                axis=axis,
                selected_id=None,
                top_probability=0.0,
                second_probability=0.0,
                margin=0.0,
                margin_threshold=margin_threshold,
                minimum_probability=self.minimum_probability,
                passed=False,
                candidate_probabilities=(),
            )
        candidate_ids = sorted(candidates)
        raw_scores = self.backend.score(
            source_text,
            prompt,
            [candidates[candidate_id] for candidate_id in candidate_ids],
        ).float()
        if raw_scores.shape != (len(candidate_ids),) or not torch.isfinite(raw_scores).all():
            return LikelihoodAxisScore(
                axis=axis,
                selected_id=None,
                top_probability=0.0,
                second_probability=0.0,
                margin=0.0,
                margin_threshold=margin_threshold,
                minimum_probability=self.minimum_probability,
                passed=False,
                candidate_probabilities=(),
            )
        probabilities = torch.softmax(raw_scores, dim=0).detach().cpu().tolist()
        ranked = sorted(
            zip(candidate_ids, probabilities), key=lambda item: (-item[1], item[0])
        )
        selected_id, top_probability = ranked[0]
        second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_probability - second_probability
        return LikelihoodAxisScore(
            axis=axis,
            selected_id=selected_id,
            top_probability=round(float(top_probability), 6),
            second_probability=round(float(second_probability), 6),
            margin=round(float(margin), 6),
            margin_threshold=margin_threshold,
            minimum_probability=self.minimum_probability,
            passed=(
                top_probability >= self.minimum_probability
                and margin >= margin_threshold
            ),
            candidate_probabilities=tuple(
                (candidate_id, round(float(probability), 6))
                for candidate_id, probability in sorted(
                    zip(candidate_ids, probabilities), key=lambda item: item[0]
                )
            ),
        )

    def produce(
        self,
        hypothesis_id: str,
        episode_id: int,
        source_text: str,
        entity_candidates: Mapping[str, str],
        attribute_candidates: Mapping[str, str],
        provenance: Tuple[str, ...] = (),
    ) -> LikelihoodProducerResult:
        """Produce and submit one conservative read-only query interpretation."""
        scores = (
            self._score_axis(
                "entity",
                source_text,
                self.entity_prompt,
                entity_candidates,
                self.entity_margin_threshold,
            ),
            self._score_axis(
                "attribute",
                source_text,
                self.attribute_prompt,
                attribute_candidates,
                self.attribute_margin_threshold,
            ),
        )
        reasons: List[str] = []
        for score in scores:
            if score.selected_id is None:
                reasons.append(f"{self.REASON_EMPTY_CANDIDATES}:{score.axis}")
            elif score.top_probability < score.minimum_probability:
                reasons.append(f"{self.REASON_BELOW_PROBABILITY}:{score.axis}")
            elif score.margin < score.margin_threshold:
                reasons.append(f"{self.REASON_MARGIN_TOO_SMALL}:{score.axis}")
        if any(not score.candidate_probabilities for score in scores):
            reasons.append(self.REASON_NONFINITE_SCORE)
        if reasons:
            return LikelihoodProducerResult(
                emitted=False,
                reason_codes=tuple(sorted(set(reasons))),
                scores=scores,
                hypothesis=None,
                adapter_decision=None,
            )

        selected = {score.axis: str(score.selected_id) for score in scores}
        confidence = min(score.top_probability for score in scores)
        hypothesis = SemanticHypothesis(
            hypothesis_id=hypothesis_id,
            episode_id=episode_id,
            mode=HypothesisMode.QUERY,
            source_text=source_text,
            producer="conservative_likelihood_query_producer",
            producer_version="1.0",
            provenance=tuple(provenance)
            + (f"backend:{self.backend.backend_id}:{self.backend.backend_version}",),
            confidence=confidence,
            uncertainty=1.0 - confidence,
            requested_destination=RequestedDestination.QUERY_ONLY,
            entity_id=selected["entity"],
            attr_type=selected["attribute"],
        )
        decision = self.adapter.submit(hypothesis)
        return LikelihoodProducerResult(
            emitted=True,
            reason_codes=(self.REASON_EMITTED,),
            scores=scores,
            hypothesis=hypothesis,
            adapter_decision=decision,
        )


class ConservativeMultiViewLikelihoodQueryProducer:
    """Fuse independent likelihood views before emitting a query interpretation."""

    REASON_EMITTED = "EMITTED"
    REASON_EMPTY_CANDIDATES = "EMPTY_CANDIDATES"
    REASON_MARGIN_TOO_SMALL = "MARGIN_TOO_SMALL"
    REASON_CONSENSUS_TOO_SMALL = "CONSENSUS_TOO_SMALL"
    REASON_NONFINITE_SCORE = "NONFINITE_SCORE"

    def __init__(
        self,
        backend: CandidateScoringBackend,
        adapter: ConservativeSemanticAdapter,
        entity_prompts: Sequence[str],
        attribute_prompts: Sequence[str],
        entity_margin_threshold: float,
        attribute_margin_threshold: float,
        entity_minimum_consensus: int,
        attribute_minimum_consensus: int,
    ) -> None:
        for name, threshold in (
            ("entity_margin_threshold", entity_margin_threshold),
            ("attribute_margin_threshold", attribute_margin_threshold),
        ):
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not entity_prompts or not attribute_prompts:
            raise ValueError("entity_prompts and attribute_prompts must not be empty")
        if any("{source_text}" not in prompt for prompt in entity_prompts):
            raise ValueError("every entity prompt must contain {source_text}")
        if any("{source_text}" not in prompt for prompt in attribute_prompts):
            raise ValueError("every attribute prompt must contain {source_text}")
        if not 1 <= entity_minimum_consensus <= len(entity_prompts):
            raise ValueError("entity_minimum_consensus is outside the view count")
        if not 1 <= attribute_minimum_consensus <= len(attribute_prompts):
            raise ValueError("attribute_minimum_consensus is outside the view count")
        self.backend = backend
        self.adapter = adapter
        self.entity_prompts = tuple(entity_prompts)
        self.attribute_prompts = tuple(attribute_prompts)
        self.entity_margin_threshold = entity_margin_threshold
        self.attribute_margin_threshold = attribute_margin_threshold
        self.entity_minimum_consensus = entity_minimum_consensus
        self.attribute_minimum_consensus = attribute_minimum_consensus

    def _score_axis(
        self,
        axis: str,
        source_text: str,
        prompts: Sequence[str],
        candidates: Mapping[str, str],
        margin_threshold: float,
        minimum_consensus: int,
    ) -> MultiViewAxisScore:
        if not candidates or any(not value.strip() for value in candidates.values()):
            return MultiViewAxisScore(
                axis=axis,
                selected_id=None,
                top_probability=0.0,
                second_probability=0.0,
                margin=0.0,
                margin_threshold=margin_threshold,
                consensus_count=0,
                minimum_consensus=minimum_consensus,
                total_views=len(prompts),
                passed=False,
                candidate_probabilities=(),
                view_votes=(),
            )
        candidate_ids = sorted(candidates)
        continuations = [candidates[candidate_id] for candidate_id in candidate_ids]
        probability_views: List[torch.Tensor] = []
        for prompt in prompts:
            raw_scores = self.backend.score(source_text, prompt, continuations).float()
            if (
                raw_scores.shape != (len(candidate_ids),)
                or not torch.isfinite(raw_scores).all()
            ):
                return MultiViewAxisScore(
                    axis=axis,
                    selected_id=None,
                    top_probability=0.0,
                    second_probability=0.0,
                    margin=0.0,
                    margin_threshold=margin_threshold,
                    consensus_count=0,
                    minimum_consensus=minimum_consensus,
                    total_views=len(prompts),
                    passed=False,
                    candidate_probabilities=(),
                    view_votes=(),
                )
            probability_views.append(torch.softmax(raw_scores, dim=0).detach().cpu())
        stacked = torch.stack(probability_views)
        aggregate = stacked.mean(dim=0)
        aggregate_values = aggregate.tolist()
        ranked = sorted(
            zip(candidate_ids, aggregate_values), key=lambda item: (-item[1], item[0])
        )
        selected_id, top_probability = ranked[0]
        second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_probability - second_probability
        view_votes = tuple(
            candidate_ids[int(view.argmax().item())] for view in probability_views
        )
        consensus_count = sum(vote == selected_id for vote in view_votes)
        return MultiViewAxisScore(
            axis=axis,
            selected_id=selected_id,
            top_probability=round(float(top_probability), 6),
            second_probability=round(float(second_probability), 6),
            margin=round(float(margin), 6),
            margin_threshold=margin_threshold,
            consensus_count=consensus_count,
            minimum_consensus=minimum_consensus,
            total_views=len(prompts),
            passed=(
                margin >= margin_threshold
                and consensus_count >= minimum_consensus
            ),
            candidate_probabilities=tuple(
                (candidate_id, round(float(probability), 6))
                for candidate_id, probability in sorted(
                    zip(candidate_ids, aggregate_values), key=lambda item: item[0]
                )
            ),
            view_votes=view_votes,
        )

    def produce(
        self,
        hypothesis_id: str,
        episode_id: int,
        source_text: str,
        entity_candidates: Mapping[str, str],
        attribute_candidates: Mapping[str, str],
        provenance: Tuple[str, ...] = (),
    ) -> MultiViewProducerResult:
        """Produce and submit one consensus-backed read-only interpretation."""
        scores = (
            self._score_axis(
                "entity",
                source_text,
                self.entity_prompts,
                entity_candidates,
                self.entity_margin_threshold,
                self.entity_minimum_consensus,
            ),
            self._score_axis(
                "attribute",
                source_text,
                self.attribute_prompts,
                attribute_candidates,
                self.attribute_margin_threshold,
                self.attribute_minimum_consensus,
            ),
        )
        reasons: List[str] = []
        for score in scores:
            if score.selected_id is None:
                reasons.append(f"{self.REASON_EMPTY_CANDIDATES}:{score.axis}")
            elif score.margin < score.margin_threshold:
                reasons.append(f"{self.REASON_MARGIN_TOO_SMALL}:{score.axis}")
            elif score.consensus_count < score.minimum_consensus:
                reasons.append(f"{self.REASON_CONSENSUS_TOO_SMALL}:{score.axis}")
        if any(not score.candidate_probabilities for score in scores):
            reasons.append(self.REASON_NONFINITE_SCORE)
        if reasons:
            return MultiViewProducerResult(
                emitted=False,
                reason_codes=tuple(sorted(set(reasons))),
                scores=scores,
                hypothesis=None,
                adapter_decision=None,
            )

        selected = {score.axis: str(score.selected_id) for score in scores}
        confidence = min(score.top_probability for score in scores)
        hypothesis = SemanticHypothesis(
            hypothesis_id=hypothesis_id,
            episode_id=episode_id,
            mode=HypothesisMode.QUERY,
            source_text=source_text,
            producer="conservative_multiview_likelihood_query_producer",
            producer_version="1.0",
            provenance=tuple(provenance)
            + (f"backend:{self.backend.backend_id}:{self.backend.backend_version}",),
            confidence=confidence,
            uncertainty=1.0 - confidence,
            requested_destination=RequestedDestination.QUERY_ONLY,
            entity_id=selected["entity"],
            attr_type=selected["attribute"],
        )
        decision = self.adapter.submit(hypothesis)
        return MultiViewProducerResult(
            emitted=True,
            reason_codes=(self.REASON_EMITTED,),
            scores=scores,
            hypothesis=hypothesis,
            adapter_decision=decision,
        )


class ConservativeTrainedQueryProducer:
    """Emit trained semantic query interpretations only above frozen margins."""

    REASON_EMITTED = "EMITTED"
    REASON_UNKNOWN_SELECTED = "UNKNOWN_SELECTED"
    REASON_MARGIN_TOO_SMALL = "MARGIN_TOO_SMALL"
    REASON_NONFINITE_SCORE = "NONFINITE_SCORE"

    def __init__(
        self,
        backend: SemanticClassificationBackend,
        adapter: ConservativeSemanticAdapter,
        entity_margin_threshold: float,
        attribute_margin_threshold: float,
    ) -> None:
        for name, threshold in (
            ("entity_margin_threshold", entity_margin_threshold),
            ("attribute_margin_threshold", attribute_margin_threshold),
        ):
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self.backend = backend
        self.adapter = adapter
        self.entity_margin_threshold = entity_margin_threshold
        self.attribute_margin_threshold = attribute_margin_threshold

    @staticmethod
    def _score_axis(
        axis: str,
        probabilities: torch.Tensor,
        candidate_ids: Sequence[str],
        unknown_id: str,
        margin_threshold: float,
    ) -> ClassificationAxisScore:
        if probabilities.shape != (len(candidate_ids),) or not torch.isfinite(
            probabilities
        ).all():
            probabilities = torch.zeros(len(candidate_ids), dtype=torch.float32)
            probabilities[candidate_ids.index(unknown_id)] = 1.0
        values = probabilities.detach().cpu().float().tolist()
        ranked = sorted(
            zip(candidate_ids, values), key=lambda item: (-item[1], item[0])
        )
        selected_id, top_probability = ranked[0]
        second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_probability - second_probability
        unknown_selected = selected_id == unknown_id
        return ClassificationAxisScore(
            axis=axis,
            selected_id=selected_id,
            top_probability=round(float(top_probability), 6),
            second_probability=round(float(second_probability), 6),
            margin=round(float(margin), 6),
            margin_threshold=margin_threshold,
            unknown_selected=unknown_selected,
            passed=(not unknown_selected and margin >= margin_threshold),
            candidate_probabilities=tuple(
                (candidate_id, round(float(probability), 6))
                for candidate_id, probability in sorted(
                    zip(candidate_ids, values), key=lambda item: item[0]
                )
            ),
        )

    def produce(
        self,
        hypothesis_id: str,
        episode_id: int,
        source_text: str,
        provenance: Tuple[str, ...] = (),
    ) -> ClassificationProducerResult:
        """Produce and submit one trained read-only semantic interpretation."""
        entity_probabilities, attribute_probabilities = self.backend.classify(
            [source_text]
        )
        scores = (
            self._score_axis(
                "entity",
                entity_probabilities[0],
                self.backend.entity_ids,
                self.backend.unknown_entity_id,
                self.entity_margin_threshold,
            ),
            self._score_axis(
                "attribute",
                attribute_probabilities[0],
                self.backend.attribute_ids,
                self.backend.unknown_attribute_id,
                self.attribute_margin_threshold,
            ),
        )
        reasons: List[str] = []
        for score in scores:
            if score.unknown_selected:
                reasons.append(f"{self.REASON_UNKNOWN_SELECTED}:{score.axis}")
            elif score.margin < score.margin_threshold:
                reasons.append(f"{self.REASON_MARGIN_TOO_SMALL}:{score.axis}")
            if not score.candidate_probabilities:
                reasons.append(f"{self.REASON_NONFINITE_SCORE}:{score.axis}")
        if reasons:
            return ClassificationProducerResult(
                emitted=False,
                reason_codes=tuple(sorted(set(reasons))),
                scores=scores,
                hypothesis=None,
                adapter_decision=None,
            )

        selected = {score.axis: score.selected_id for score in scores}
        confidence = min(score.top_probability for score in scores)
        hypothesis = SemanticHypothesis(
            hypothesis_id=hypothesis_id,
            episode_id=episode_id,
            mode=HypothesisMode.QUERY,
            source_text=source_text,
            producer="conservative_trained_query_producer",
            producer_version="1.0",
            provenance=tuple(provenance)
            + (f"backend:{self.backend.backend_id}:{self.backend.backend_version}",),
            confidence=confidence,
            uncertainty=1.0 - confidence,
            requested_destination=RequestedDestination.QUERY_ONLY,
            entity_id=selected["entity"],
            attr_type=selected["attribute"],
        )
        decision = self.adapter.submit(hypothesis)
        return ClassificationProducerResult(
            emitted=True,
            reason_codes=(self.REASON_EMITTED,),
            scores=scores,
            hypothesis=hypothesis,
            adapter_decision=decision,
        )
