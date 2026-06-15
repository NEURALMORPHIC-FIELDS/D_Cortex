# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b-R frozen read-only bridge readiness evaluation.

import argparse
import ast
import contextlib
import gc
import hashlib
import io
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
for _extra in (REPO_ROOT, REPO_ROOT / "colab"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import torch
import tiktoken

from dcortex.model import DCortexV2Model
from dcortex.semantic_adapter import (
    AdapterDecision,
    ConservativeSemanticAdapter,
    DecisionStatus,
    HypothesisMode,
    RequestedDestination,
    SemanticHypothesis,
)
from dcortex.semantic_producer import (
    ConservativeTrainedQueryProducer,
    DCortexContextualFeatureBackend,
    PooledSemanticClassificationBackend,
    SemanticQueryHead,
)
from dcortex.semantic_query_bridge import (
    QueryRouteStatus,
    ReadOnlyQueryRoute,
    ReadOnlySemanticQueryBridge,
)
from scripts.semantic_contextual_curriculum import load_contextual_model
from scripts.semantic_likelihood_probe import SEALED_SHA256, SEALED_SOURCE, gate
from scripts.train_semantic_internalizer import (
    ATTRIBUTE_MARGIN,
    ENTITY_MARGIN,
    UNKNOWN_ATTRIBUTE,
    UNKNOWN_ENTITY,
)
from train_campaign import big_config

SEP = "=" * 70
ENC = tiktoken.get_encoding("gpt2")
EOT = ENC.eot_token
SEQ_LEN = 64
LEXICAL_ALPHA = 0.9
TRIALS_PER_FAMILY = 200
SAMPLE_SEED = 20261315
FAMILIES = ("F1", "F3", "F5", "S5", "S6")

ADAPTER_SOURCE = REPO_ROOT / "dcortex" / "semantic_adapter.py"
PRODUCER_SOURCE = REPO_ROOT / "dcortex" / "semantic_producer.py"
CONTEXTUAL_SOURCE = REPO_ROOT / "scripts" / "semantic_contextual_curriculum.py"
FROZEN_HASHES = {
    "pas7a": SEALED_SHA256,
    "semantic_adapter": "719afbe8359af4ee98006e0fa19f169a0a40f22ee0cc41e5dc1459d6e07e605e",
    "semantic_producer": "24797a121c90f78161518362f0401bbc1b161546e87bac194adb4144f4aa67d0",
    "semantic_contextual_evaluator": "bab8ec1268d5000d5a9da5ae5946d29946822efa9a6b0b3ce53be59052631b57",
}


@dataclass
class BridgeEpisode:
    """One frozen bridge-readiness trial."""

    family: str
    index: int
    query: str
    entity: Optional[str]
    attribute: str
    target_value: Optional[str]
    facts: Tuple[Tuple[str, str], ...]
    ignored_texts: Tuple[str, ...]
    target_unknown: bool
    form_index: Optional[int]
    hypothesis: Optional[SemanticHypothesis] = None
    decision: Optional[AdapterDecision] = None
    route: Optional[ReadOnlyQueryRoute] = None

    def sample_dict(self) -> Dict[str, Any]:
        """Return deterministic sample-only data."""
        return {
            "family": self.family,
            "index": self.index,
            "query": self.query,
            "entity": self.entity,
            "attribute": self.attribute,
            "target_value": self.target_value,
            "facts": [list(item) for item in self.facts],
            "ignored_texts": list(self.ignored_texts),
            "target_unknown": self.target_unknown,
            "form_index": self.form_index,
        }


@contextlib.contextmanager
def silent() -> Any:
    """Suppress verbose model lifecycle output."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def sha256_file(path: Path) -> str:
    """Return lowercase SHA-256 for one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_definitions() -> Dict[str, Any]:
    """Extract only frozen Pas 7a sample definitions without executing it."""
    required = {
        "HOLDOUT_ENTITIES_SINGLE",
        "HOLDOUT_COLORS",
        "HOLDOUT_SIZES",
        "HOLDOUT_LOCATIONS",
        "HOLDOUT_STATES",
        "HOLDOUT_ATTR_VALUES",
        "HOLDOUT_ATTR_TYPES",
        "F1_FACT_CONSTRUCTIONS",
        "F1_QUERY_CONSTRUCTIONS",
        "F3_NOVEL_ALIAS_QUERIES",
        "F4_DISTRACTOR_SENTENCES",
        "F5_QUERY_FORMS",
    }
    tree = ast.parse(SEALED_SOURCE.read_text(encoding="utf-8"))
    selected: List[ast.stmt] = []
    found = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = {
            target.id for target in node.targets if isinstance(target, ast.Name)
        }
        if names.intersection(required):
            selected.append(node)
            found.update(names.intersection(required))
    missing = required - found
    if missing:
        raise RuntimeError(f"Missing sealed definitions: {sorted(missing)}")
    namespace: Dict[str, Any] = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SEALED_SOURCE), "exec"), namespace)
    return {name: namespace[name] for name in required}


def sample_hash(episodes: Sequence[BridgeEpisode]) -> str:
    """Return deterministic hash of a complete sample."""
    payload = json.dumps(
        [episode.sample_dict() for episode in episodes],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_query(entity: str, attribute: str) -> str:
    """Return the standard query used by S5/S6 controls."""
    return ReadOnlySemanticQueryBridge.CANONICAL_READ_TEMPLATES[attribute].format(
        entity=entity
    )


def build_sample(definitions: Mapping[str, Any]) -> List[BridgeEpisode]:
    """Build the frozen F1/F3/F5/S5/S6 sample."""
    rng = random.Random(SAMPLE_SEED)
    entities = definitions["HOLDOUT_ENTITIES_SINGLE"]
    attributes = definitions["HOLDOUT_ATTR_TYPES"]
    values = definitions["HOLDOUT_ATTR_VALUES"]
    distractors = definitions["F4_DISTRACTOR_SENTENCES"]
    episodes: List[BridgeEpisode] = []
    for family in FAMILIES:
        for index in range(TRIALS_PER_FAMILY):
            attribute = rng.choice(attributes)
            if family == "F1":
                entity = rng.choice(entities)
                value = rng.choice(values[attribute])
                fact_form = rng.randrange(4)
                query_form = rng.randrange(4)
                facts = [
                    (
                        definitions["F1_FACT_CONSTRUCTIONS"][attribute][fact_form](
                            entity, value
                        ),
                        value,
                    )
                ]
                for _ in range(rng.choice((1, 2))):
                    other_entity = rng.choice([item for item in entities if item != entity])
                    other_attribute = rng.choice(attributes)
                    other_value = rng.choice(values[other_attribute])
                    other_form = rng.randrange(4)
                    facts.append(
                        (
                            definitions["F1_FACT_CONSTRUCTIONS"][other_attribute][
                                other_form
                            ](other_entity, other_value),
                            other_value,
                        )
                    )
                rng.shuffle(facts)
                query = definitions["F1_QUERY_CONSTRUCTIONS"][attribute][query_form](
                    entity
                )
                episodes.append(
                    BridgeEpisode(
                        family=family,
                        index=index,
                        query=query,
                        entity=entity,
                        attribute=attribute,
                        target_value=value,
                        facts=tuple(facts),
                        ignored_texts=(),
                        target_unknown=False,
                        form_index=query_form,
                    )
                )
            elif family == "F3":
                entity = rng.choice(entities)
                value = rng.choice(values[attribute])
                form_index = rng.randrange(4)
                query = definitions["F3_NOVEL_ALIAS_QUERIES"][attribute][form_index](
                    entity
                )
                episodes.append(
                    BridgeEpisode(
                        family=family,
                        index=index,
                        query=query,
                        entity=entity,
                        attribute=attribute,
                        target_value=value,
                        facts=((f"The {entity} is {value}.", value),),
                        ignored_texts=(),
                        target_unknown=False,
                        form_index=form_index,
                    )
                )
            elif family == "F5":
                entity = rng.choice(entities)
                value = rng.choice(values[attribute])
                form_index = rng.randrange(4)
                query = definitions["F5_QUERY_FORMS"][attribute][form_index](
                    entity, value
                )
                episodes.append(
                    BridgeEpisode(
                        family=family,
                        index=index,
                        query=query,
                        entity=entity,
                        attribute=attribute,
                        target_value=value,
                        facts=((f"The {entity} is {value}.", value),),
                        ignored_texts=(),
                        target_unknown=False,
                        form_index=form_index,
                    )
                )
            elif family == "S5":
                entity = rng.choice(entities)
                value_a, value_b = rng.sample(values[attribute], 2)
                ignored = tuple(rng.sample(distractors, 2))
                episodes.append(
                    BridgeEpisode(
                        family=family,
                        index=index,
                        query=canonical_query(entity, attribute),
                        entity=entity,
                        attribute=attribute,
                        target_value=None,
                        facts=(
                            (f"The {entity} is {value_a}.", value_a),
                            (f"The {entity} is {value_b}.", value_b),
                        ),
                        ignored_texts=ignored,
                        target_unknown=True,
                        form_index=None,
                    )
                )
            else:
                entity_a, entity_b = rng.sample(entities, 2)
                value_a = rng.choice(values[attribute])
                value_b = rng.choice(values[attribute])
                pronoun_query = {
                    "color": "What color is it? It is",
                    "size": "What size is it? It is",
                    "location": "Where is it? It is in the",
                    "state": "What state is it in? It is",
                }[attribute]
                episodes.append(
                    BridgeEpisode(
                        family=family,
                        index=index,
                        query=pronoun_query,
                        entity=None,
                        attribute=attribute,
                        target_value=None,
                        facts=(
                            (f"The {entity_a} is {value_a}.", value_a),
                            (f"The {entity_b} is {value_b}.", value_b),
                        ),
                        ignored_texts=(rng.choice(distractors),),
                        target_unknown=True,
                        form_index=None,
                    )
                )
    return episodes


def load_semantic_backend(
    feature_backend: DCortexContextualFeatureBackend,
    head_path: Path,
    device: torch.device,
) -> PooledSemanticClassificationBackend:
    """Load one sealed contextual semantic head."""
    payload = torch.load(head_path, map_location="cpu", weights_only=False)
    head = SemanticQueryHead(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        entity_classes=len(payload["entity_ids"]),
        attribute_classes=len(payload["attribute_ids"]),
        dropout=0.1,
    ).to(device)
    head.load_state_dict(payload["head"], strict=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return PooledSemanticClassificationBackend(
        feature_backend=feature_backend,
        head=head,
        entity_ids=payload["entity_ids"],
        attribute_ids=payload["attribute_ids"],
        unknown_entity_id=UNKNOWN_ENTITY,
        unknown_attribute_id=UNKNOWN_ATTRIBUTE,
        backend_version=head_path.name,
    )


def route_batch(
    episodes: Sequence[BridgeEpisode],
    backend: PooledSemanticClassificationBackend,
    bridge: ReadOnlySemanticQueryBridge,
    provenance: str,
) -> None:
    """Produce adapter-approved routes from one semantic backend in a batch."""
    if not episodes:
        return
    entity_probabilities, attribute_probabilities = backend.classify(
        [episode.query for episode in episodes]
    )
    for row, episode in enumerate(episodes):
        scores = (
            ConservativeTrainedQueryProducer._score_axis(
                "entity",
                entity_probabilities[row],
                backend.entity_ids,
                backend.unknown_entity_id,
                ENTITY_MARGIN,
            ),
            ConservativeTrainedQueryProducer._score_axis(
                "attribute",
                attribute_probabilities[row],
                backend.attribute_ids,
                backend.unknown_attribute_id,
                ATTRIBUTE_MARGIN,
            ),
        )
        if all(score.passed for score in scores):
            selected = {score.axis: score.selected_id for score in scores}
            confidence = min(score.top_probability for score in scores)
            hypothesis = SemanticHypothesis(
                hypothesis_id=f"{episode.family}-{episode.index}",
                episode_id=episode.index,
                mode=HypothesisMode.QUERY,
                source_text=episode.query,
                producer="conservative_trained_query_producer",
                producer_version="1.0",
                provenance=(provenance, f"trial:{episode.index}"),
                confidence=confidence,
                uncertainty=1.0 - confidence,
                requested_destination=RequestedDestination.QUERY_ONLY,
                entity_id=selected["entity"],
                attr_type=selected["attribute"],
            )
            decision = ConservativeSemanticAdapter().submit(hypothesis)
        else:
            hypothesis = None
            decision = None
        episode.hypothesis = hypothesis
        episode.decision = decision
        episode.route = bridge.route(episode.query, hypothesis, decision)


def build_routes(
    episodes: Sequence[BridgeEpisode],
    checkpoint: Path,
    heads_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Build all frozen semantic routes, then release the contextual substrate."""
    model, state, payload = load_contextual_model(checkpoint, device)
    feature_backend = DCortexContextualFeatureBackend(
        model=model,
        tokenizer=ENC.encode_ordinary,
        max_seq_len=128,
        batch_size=128,
        backend_version=checkpoint.name,
    )
    bridge = ReadOnlySemanticQueryBridge()
    backends = {
        f"fold_{fold}": load_semantic_backend(
            feature_backend, heads_dir / f"fold_{fold}_best_head.pt", device
        )
        for fold in range(4)
    }
    backends["final"] = load_semantic_backend(
        feature_backend, heads_dir / "final_best_head.pt", device
    )
    for family in ("F1", "F3"):
        for fold in range(4):
            route_batch(
                [
                    episode
                    for episode in episodes
                    if episode.family == family and episode.form_index == fold
                ],
                backends[f"fold_{fold}"],
                bridge,
                f"family:{family}:heldout_fold:{fold}",
            )
    for family in ("F5", "S5", "S6"):
        route_batch(
            [episode for episode in episodes if episode.family == family],
            backends["final"],
            bridge,
            f"family:{family}:final_head",
        )
    substrate_hash = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        substrate_hash.update(key.encode("utf-8"))
        substrate_hash.update(tensor.numpy().tobytes())
    result = {
        "substrate_state_hash": substrate_hash.hexdigest(),
        "checkpoint_mtime": datetime.fromtimestamp(checkpoint.stat().st_mtime)
        .astimezone()
        .isoformat(timespec="seconds"),
    }
    del backends, feature_backend, model, state, payload
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def pad(ids: List[int]) -> List[int]:
    """Pad or truncate a sequence to the memory campaign width."""
    return ids[:SEQ_LEN] if len(ids) > SEQ_LEN else ids + [EOT] * (SEQ_LEN - len(ids))


def memory_state_hash(model: DCortexV2Model) -> str:
    """Hash all memory-related buffers and differentiable overlays."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.named_buffers(), key=lambda item: item[0]):
        if (
            "_mem." not in name
            and not name.startswith("encoder.episode_ssm.")
            and name != "step_counter"
        ):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    for bank_name, bank in sorted(model._bank_dict().items()):
        for slot in sorted(bank._overlay):
            for key, tensor in sorted(bank._overlay[slot].items()):
                value = tensor.detach().cpu().contiguous()
                digest.update(f"{bank_name}:{slot}:{key}".encode("utf-8"))
                digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_memory_model(checkpoint: Path, device: torch.device) -> DCortexV2Model:
    """Load the trained neural-memory model used by the frozen evaluation."""
    if not checkpoint.exists():
        raise RuntimeError(f"Memory checkpoint not found: {checkpoint}")
    with silent():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model = DCortexV2Model(big_config()).to(device)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
    del payload
    return model


@torch.inference_mode()
def write_episode(model: DCortexV2Model, episode: BridgeEpisode, device: torch.device) -> None:
    """Write the frozen facts to the neural working bank."""
    with silent():
        model.reset_memory()
        model.begin_episode()
    for fact_text, value in episode.facts:
        fact_ids = pad(ENC.encode_ordinary(fact_text) + [EOT])
        answer_ids = ENC.encode_ordinary(" " + value)
        if not answer_ids:
            raise RuntimeError(f"Value tokenization failed: {value}")
        fact_tensor = torch.tensor([fact_ids], dtype=torch.long, device=device)
        answer_tensor = torch.tensor([answer_ids[0]], dtype=torch.long, device=device)
        with silent():
            model.encode(
                fact_tensor,
                answer_token_id=answer_tensor,
                lexical_alpha=LEXICAL_ALPHA,
                force_bank="working",
            )


@torch.inference_mode()
def read_answer(model: DCortexV2Model, query: str, device: torch.device) -> int:
    """Read one first-token answer from neural memory."""
    query_tensor = torch.tensor(
        [pad(ENC.encode_ordinary(query))], dtype=torch.long, device=device
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        _, retrieved = model.decode(query_tensor, return_retrieved=True)
        answer = model.aux_answer_head(retrieved).float()
    return int(answer[0].argmax().item())


def summarize_routes(episodes: Sequence[BridgeEpisode]) -> Dict[str, Dict[str, Any]]:
    """Summarize semantic route quality per family."""
    output: Dict[str, Dict[str, Any]] = {}
    for family in FAMILIES:
        subset = [episode for episode in episodes if episode.family == family]
        routed = sum(
            episode.route is not None and episode.route.status == QueryRouteStatus.ROUTED
            for episode in subset
        )
        correct = sum(
            episode.route is not None
            and episode.route.status == QueryRouteStatus.ROUTED
            and episode.route.entity_id == episode.entity
            and episode.route.attr_type == episode.attribute
            for episode in subset
        )
        wrong = routed - correct if family in ("F1", "F3", "F5") else None
        output[family] = {
            "n": len(subset),
            "routed": routed,
            "route_rate": routed / len(subset),
            "correct": correct if family in ("F1", "F3", "F5") else None,
            "accuracy": correct / len(subset) if family in ("F1", "F3", "F5") else None,
            "wrong": wrong,
            "wrong_rate": wrong / len(subset) if wrong is not None else None,
            "fallback": len(subset) - routed,
        }
    return output


def run_memory_evaluation(
    episodes: Sequence[BridgeEpisode],
    model: DCortexV2Model,
    device: torch.device,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Run baseline and routed reads over identical neural-memory writes."""
    counters = {
        family: {
            "n": 0,
            "baseline_correct": 0,
            "routed_correct": 0,
            "baseline_overcommit": 0,
            "routed_overcommit": 0,
        }
        for family in FAMILIES
    }
    details: List[Dict[str, Any]] = []
    bridge_invariant = True
    read_invariant = True
    bridge = ReadOnlySemanticQueryBridge()
    for position, episode in enumerate(episodes):
        write_episode(model, episode, device)
        post_write_hash = memory_state_hash(model)
        repeated_route = bridge.route(episode.query, episode.hypothesis, episode.decision)
        post_bridge_hash = memory_state_hash(model)
        bridge_invariant = bridge_invariant and post_write_hash == post_bridge_hash
        baseline_prediction = read_answer(model, episode.query, device)
        post_baseline_hash = memory_state_hash(model)
        assert episode.route is not None
        routed_prediction = read_answer(model, episode.route.routed_query, device)
        post_routed_hash = memory_state_hash(model)
        read_invariant = read_invariant and (
            post_bridge_hash == post_baseline_hash == post_routed_hash
        )
        family_counter = counters[episode.family]
        family_counter["n"] += 1
        target_token = (
            None
            if episode.target_unknown or episode.target_value is None
            else ENC.encode_ordinary(" " + episode.target_value)[0]
        )
        if target_token is not None:
            family_counter["baseline_correct"] += int(
                baseline_prediction == target_token
            )
            family_counter["routed_correct"] += int(routed_prediction == target_token)
        else:
            family_counter["baseline_overcommit"] += int(baseline_prediction != EOT)
            family_counter["routed_overcommit"] += int(routed_prediction != EOT)
        details.append(
            {
                "position": position,
                "family": episode.family,
                "trial": episode.index,
                "query": episode.query,
                "route_status": episode.route.status.value,
                "routed_query": episode.route.routed_query,
                "semantic_entity": episode.route.entity_id,
                "semantic_attribute": episode.route.attr_type,
                "expected_entity": episode.entity,
                "expected_attribute": episode.attribute,
                "target_token": target_token,
                "baseline_prediction": baseline_prediction,
                "routed_prediction": routed_prediction,
                "bridge_state_unchanged": post_write_hash == post_bridge_hash,
                "baseline_read_state_unchanged": post_bridge_hash == post_baseline_hash,
                "routed_read_state_unchanged": post_baseline_hash == post_routed_hash,
                "route_repeat_exact": repeated_route.to_json() == episode.route.to_json(),
            }
        )
        if (position + 1) % 50 == 0:
            print(
                f"[INFO] Memory evaluation {position + 1}/{len(episodes)}",
                flush=True,
            )
    metrics: Dict[str, Dict[str, Any]] = {}
    for family, values in counters.items():
        n = values["n"]
        if family in ("F1", "F3", "F5"):
            baseline_recall = values["baseline_correct"] / n
            routed_recall = values["routed_correct"] / n
            metrics[family] = {
                **values,
                "baseline_recall": baseline_recall,
                "routed_recall": routed_recall,
                "uplift": routed_recall - baseline_recall,
                "baseline_wrong_rate": 1.0 - baseline_recall,
                "routed_wrong_rate": 1.0 - routed_recall,
            }
        else:
            baseline_overcommit = values["baseline_overcommit"] / n
            routed_overcommit = values["routed_overcommit"] / n
            metrics[family] = {
                **values,
                "baseline_honesty": 1.0 - baseline_overcommit,
                "baseline_overcommit_rate": baseline_overcommit,
                "routed_honesty": 1.0 - routed_overcommit,
                "routed_overcommit_rate": routed_overcommit,
            }
    return metrics, details, {
        "bridge_state_unchanged_all": bridge_invariant,
        "read_state_unchanged_all": read_invariant,
    }


def run(args: argparse.Namespace) -> int:
    """Run the frozen bridge readiness verdict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Frozen bridge end-to-end evaluation requires CUDA")
    results_dir = Path(args.run_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    definitions = extract_definitions()
    episodes_a = build_sample(definitions)
    episodes_b = build_sample(definitions)
    sample_hash_a = sample_hash(episodes_a)
    sample_hash_b = sample_hash(episodes_b)

    print(SEP, flush=True)
    print("[INFO] Building frozen semantic routes", flush=True)
    route_reference = build_routes(
        episodes_a,
        Path(args.query_checkpoint),
        Path(args.heads_dir),
        device,
    )
    route_metrics = summarize_routes(episodes_a)
    bridge = ReadOnlySemanticQueryBridge()
    all_routed_accepted_query = all(
        episode.route is not None
        and (
            episode.route.status != QueryRouteStatus.ROUTED
            or (
                episode.decision is not None
                and episode.decision.status == DecisionStatus.ACCEPT_QUERY
                and episode.hypothesis is not None
                and episode.hypothesis.mode == HypothesisMode.QUERY
                and episode.hypothesis.requested_destination
                == RequestedDestination.QUERY_ONLY
                and episode.hypothesis.hypothesis_id
                == episode.decision.hypothesis_id
            )
        )
        for episode in episodes_a
    )
    all_fallback_exact = all(
        episode.route is not None
        and (
            episode.route.status != QueryRouteStatus.FALLBACK
            or episode.route.routed_query == episode.query
        )
        for episode in episodes_a
    )
    all_route_repeat_exact = all(
        episode.route is not None
        and bridge.route(episode.query, episode.hypothesis, episode.decision).to_json()
        == episode.route.to_json()
        for episode in episodes_a
    )
    mismatch_hypothesis = next(
        episode.hypothesis for episode in episodes_a if episode.hypothesis is not None
    )
    mismatch_decision = AdapterDecision(
        status=DecisionStatus.ACCEPT_QUERY,
        hypothesis_id="mismatch",
        reason_codes=("ACCEPTED_QUERY",),
        audit_sequence=1,
    )
    mismatch_route = bridge.route(
        mismatch_hypothesis.source_text, mismatch_hypothesis, mismatch_decision
    )
    mismatch_rejected = (
        mismatch_route.status == QueryRouteStatus.FALLBACK
        and mismatch_route.routed_query == mismatch_hypothesis.source_text
    )
    forbidden_attributes = (
        "model",
        "memory",
        "bank",
        "reader",
        "writer",
        "commit",
        "provisional",
        "consolidator",
    )
    no_mutation_api = not any(hasattr(bridge, name) for name in forbidden_attributes)

    print(SEP, flush=True)
    print("[INFO] Loading trained neural-memory model", flush=True)
    memory_model = load_memory_model(Path(args.memory_checkpoint), device)
    memory_metrics, details, state_checks = run_memory_evaluation(
        episodes_a, memory_model, device
    )
    del memory_model
    gc.collect()
    torch.cuda.empty_cache()

    actual_hashes = {
        "pas7a": sha256_file(SEALED_SOURCE),
        "semantic_adapter": sha256_file(ADAPTER_SOURCE),
        "semantic_producer": sha256_file(PRODUCER_SOURCE),
        "semantic_contextual_evaluator": sha256_file(CONTEXTUAL_SOURCE),
    }
    seals_untouched = actual_hashes == FROZEN_HASHES
    semantic_quality_pass = all(
        route_metrics[family]["accuracy"] >= 0.85
        and route_metrics[family]["wrong_rate"] <= 0.02
        for family in ("F1", "F3", "F5")
    )
    uplift_pass = all(
        memory_metrics[family]["routed_recall"] >= 0.75
        and memory_metrics[family]["uplift"] >= 0.20
        for family in ("F1", "F3", "F5")
    )
    no_harm_pass = all(
        memory_metrics[family]["routed_wrong_rate"]
        <= memory_metrics[family]["baseline_wrong_rate"] + 0.02
        for family in ("F1", "F3", "F5")
    )
    honesty_pass = all(
        memory_metrics[family]["routed_honesty"] >= 0.95
        and memory_metrics[family]["routed_overcommit_rate"] <= 0.02
        and memory_metrics[family]["routed_honesty"]
        >= memory_metrics[family]["baseline_honesty"] - 0.02
        for family in ("S5", "S6")
    )
    gates = [
        gate(
            "B0_SEALS_UNTOUCHED",
            seals_untouched,
            f"All four frozen source hashes unchanged={seals_untouched}.",
            {"frozen": FROZEN_HASHES, "actual": actual_hashes},
        ),
        gate(
            "B1_ACCEPTED_QUERY_ONLY",
            all_routed_accepted_query,
            f"Every routed result has matching adapter ACCEPT_QUERY={all_routed_accepted_query}.",
            {"all_routed_accepted_query": all_routed_accepted_query},
        ),
        gate(
            "B2_NO_MUTATION_API",
            no_mutation_api,
            f"Bridge runtime has no forbidden mutation dependency={no_mutation_api}.",
            {"forbidden_attributes": list(forbidden_attributes)},
        ),
        gate(
            "B3_FALLBACK_EQUIVALENCE",
            all_fallback_exact,
            f"All fallback queries preserve exact original text={all_fallback_exact}.",
            {"all_fallback_exact": all_fallback_exact},
        ),
        gate(
            "B4_DETERMINISTIC_ROUTE",
            all_route_repeat_exact,
            f"Repeated route JSON byte-identical={all_route_repeat_exact}.",
            {"all_route_repeat_exact": all_route_repeat_exact},
        ),
        gate(
            "B5_MISMATCH_REJECTED",
            mismatch_rejected,
            f"Decision/hypothesis mismatch falls back exactly={mismatch_rejected}.",
            {"mismatch_route": mismatch_route.to_dict()},
        ),
        gate(
            "B6_SEMANTIC_ROUTE_QUALITY",
            semantic_quality_pass,
            "F1/F3/F5 route accuracy >=85% and wrong routing <=2%="
            f"{semantic_quality_pass}.",
            {family: route_metrics[family] for family in ("F1", "F3", "F5")},
        ),
        gate(
            "B7_END_TO_END_UPLIFT",
            uplift_pass,
            "Each F1/F3/F5 routed neural-memory recall >=75% and uplift >=20pp="
            f"{uplift_pass}.",
            {family: memory_metrics[family] for family in ("F1", "F3", "F5")},
        ),
        gate(
            "B8_NO_FAMILY_HARM",
            no_harm_pass,
            f"Routed wrong-answer rate adds <=2pp in every knowable family={no_harm_pass}.",
            {family: memory_metrics[family] for family in ("F1", "F3", "F5")},
        ),
        gate(
            "B9_FACT_WRITE_INVARIANCE",
            state_checks["bridge_state_unchanged_all"],
            "Bridge invocation leaves every post-write memory state byte-identical="
            f"{state_checks['bridge_state_unchanged_all']}.",
            state_checks,
        ),
        gate(
            "B10_READ_ONLY_STATE",
            state_checks["read_state_unchanged_all"],
            "Baseline and routed reads leave every memory state byte-identical="
            f"{state_checks['read_state_unchanged_all']}.",
            state_checks,
        ),
        gate(
            "B11_S5_S6_HONESTY",
            honesty_pass,
            f"S5/S6 routed honesty >=95% and overcommit <=2%={honesty_pass}.",
            {family: memory_metrics[family] for family in ("S5", "S6")},
        ),
        gate(
            "B12_SEALED_SAMPLE",
            sample_hash_a == sample_hash_b
            and len(episodes_a) == len(FAMILIES) * TRIALS_PER_FAMILY,
            f"Repeated frozen sample exact={sample_hash_a == sample_hash_b}; "
            f"count={len(episodes_a)}.",
            {
                "sample_hash_a": sample_hash_a,
                "sample_hash_b": sample_hash_b,
                "count": len(episodes_a),
                "trials_per_family": TRIALS_PER_FAMILY,
                "seed": SAMPLE_SEED,
            },
        ),
    ]
    all_pass = all(item["passed"] for item in gates)
    verdict = {
        "verdict": gates,
        "reference": {
            "run_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "device": str(device),
            "memory_checkpoint": str(Path(args.memory_checkpoint)),
            "memory_checkpoint_mtime": datetime.fromtimestamp(
                Path(args.memory_checkpoint).stat().st_mtime
            )
            .astimezone()
            .isoformat(timespec="seconds"),
            "query_checkpoint": str(Path(args.query_checkpoint)),
            "heads_dir": str(Path(args.heads_dir)),
            "route_reference": route_reference,
            "route_metrics": route_metrics,
            "memory_metrics": memory_metrics,
            "state_checks": state_checks,
            "sample_hash": sample_hash_a,
            "details": details,
            "scope": (
                "Read-only semantic bridge readiness on the locally executable "
                "trained neural working-memory path. Not Pas 7a committed/"
                "provisional integration and not a wrong-commit measurement."
            ),
            "claim_status": "MEASURED in one local environment; not PROVEN.",
            "all_pass": all_pass,
        },
    }
    verdict_path = results_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(SEP, flush=True)
    print("[INFO] D_Cortex v15.7b-R bridge readiness verdict", flush=True)
    for item in gates:
        label = "✓ PASS" if item["passed"] else "✗ FAIL"
        print(f"{label} [{item['criterion_id']}] {item['evidence']}", flush=True)
    print(SEP, flush=True)
    for family in FAMILIES:
        print(
            f"[INFO] {family} route={route_metrics[family]['route_rate']:.1%} "
            f"memory={json.dumps(memory_metrics[family], sort_keys=True)}",
            flush=True,
        )
    print(f"[INFO] Overall: {'ALL GATES PASS' if all_pass else 'GATE FAILURE'}", flush=True)
    print(f"[INFO] Verdict: {verdict_path}", flush=True)
    return 0 if all_pass else 1


def build_argparser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    source_root = REPO_ROOT.parent / "D_Cortex-main"
    parser = argparse.ArgumentParser(description="D_Cortex read-only bridge verdict")
    parser.add_argument(
        "--memory-checkpoint",
        default=str(source_root / "runs" / "memory_campaign" / "results" / "best_model.pt"),
    )
    parser.add_argument(
        "--query-checkpoint",
        default=str(source_root / "runs" / "warmstart" / "warmstarted_init.pt"),
    )
    parser.add_argument(
        "--heads-dir",
        default=str(REPO_ROOT / "runs" / "semantic_contextual" / "results"),
    )
    parser.add_argument(
        "--run-dir",
        default=str(REPO_ROOT / "runs" / "semantic_bridge_end_to_end"),
    )
    return parser


def main() -> int:
    """CLI entry point."""
    return run(build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
