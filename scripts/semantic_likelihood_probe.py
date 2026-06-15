# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b decoder-native causal-likelihood semantic producer probe.

import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import tiktoken

from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
from dcortex.semantic_adapter import (
    ConservativeSemanticAdapter,
    DecisionStatus,
    HypothesisMode,
    RequestedDestination,
)
from dcortex.semantic_producer import (
    ConservativeLikelihoodQueryProducer,
    ConservativePrototypeProducer,
    DCortexCausalLikelihoodBackend,
    DCortexTokenEmbeddingBackend,
    LikelihoodProducerResult,
)

SEP = "=" * 70
ENC = tiktoken.get_encoding("gpt2")
EOT = ENC.eot_token
SEALED_SOURCE = REPO_ROOT / "steps" / "13_v15_7a_consolidation" / "code.py"
SEALED_SHA256 = "25b4906ecc05a6b51b10902e54332a0ec2b26c4c622aa4e6ee74bd4961369aa3"

ENTITY_CANDIDATES = {
    "dragon": "dragon",
    "mermaid": "mermaid",
    "phoenix": "phoenix",
    "robot": "robot",
}
ATTRIBUTE_CANDIDATES = {
    "color": "color",
    "location": "location",
    "size": "size",
    "state": "state",
}
ENTITY_PROMPT = "Question: {source_text}\nAnswer entity:"
ATTRIBUTE_PROMPT = "Question: {source_text}\nAnswer type:"

TOKEN_MEAN_ENTITY_PROTOTYPES = {
    candidate_id: (candidate_text,)
    for candidate_id, candidate_text in ENTITY_CANDIDATES.items()
}
TOKEN_MEAN_ATTRIBUTE_PROTOTYPES = {
    "color": ("What color is the object?",),
    "location": ("Where is the object located?",),
    "size": ("What size is the object?",),
    "state": ("What state is the object in?",),
}

# Iteration-1 observed data. It is calibration/development data only.
CALIBRATION_QUERIES: Tuple[Tuple[str, str, str], ...] = (
    ("Which hue belongs to the dragon?", "dragon", "color"),
    ("Describe the phoenix's coloration.", "phoenix", "color"),
    ("What shade characterizes the mermaid?", "mermaid", "color"),
    ("Identify the robot's chromatic appearance.", "robot", "color"),
    ("How would you describe the dragon's hue?", "dragon", "color"),
    ("How large is the dragon?", "dragon", "size"),
    ("What are the phoenix's dimensions?", "phoenix", "size"),
    ("Describe the mermaid's magnitude.", "mermaid", "size"),
    ("Assess the robot's physical scale.", "robot", "size"),
    ("How big is the phoenix?", "phoenix", "size"),
    ("Where can the dragon be found?", "dragon", "location"),
    ("What place contains the phoenix?", "phoenix", "location"),
    ("Identify the mermaid's whereabouts.", "mermaid", "location"),
    ("In which place is the robot situated?", "robot", "location"),
    ("Where does the dragon reside?", "dragon", "location"),
    ("What condition is the dragon in?", "dragon", "state"),
    ("Describe the phoenix's current status.", "phoenix", "state"),
    ("What is the mermaid's present condition?", "mermaid", "state"),
    ("Identify the robot's current status.", "robot", "state"),
    ("How is the phoenix doing right now?", "phoenix", "state"),
)
CALIBRATION_AMBIGUOUS: Tuple[str, ...] = (
    "Tell me about the dragon.",
    "What do we know about the phoenix?",
    "Describe the mermaid.",
    "Give information about the robot.",
    "What is notable about the dragon?",
    "Report something concerning the phoenix.",
    "What can be said regarding the mermaid?",
    "Summarize the robot.",
    "Provide details about the dragon.",
    "What is known about the phoenix?",
)

# Frozen before the first causal-likelihood verdict run.
EVALUATION_QUERIES: Tuple[Tuple[str, str, str], ...] = (
    ("Which tint would best characterize the robot?", "robot", "color"),
    ("Describe the mermaid's visible hue.", "mermaid", "color"),
    ("What chromatic quality does the phoenix display?", "phoenix", "color"),
    ("The dragon appears in what shade?", "dragon", "color"),
    ("How is the robot pigmented?", "robot", "color"),
    ("Which tone is associated with the mermaid's appearance?", "mermaid", "color"),
    ("Name the phoenix's visible coloration.", "phoenix", "color"),
    ("What palette describes the dragon?", "dragon", "color"),
    ("Whereabouts should one seek the robot?", "robot", "location"),
    ("In what place would the dragon be encountered?", "dragon", "location"),
    ("Where is the mermaid typically found?", "mermaid", "location"),
    ("Which setting contains the phoenix?", "phoenix", "location"),
    ("Name the robot's usual whereabouts.", "robot", "location"),
    ("At what site does the dragon reside?", "dragon", "location"),
    ("Where could one find the mermaid?", "mermaid", "location"),
    ("Identify the surroundings occupied by the phoenix.", "phoenix", "location"),
    ("How enormous is the dragon?", "dragon", "size"),
    ("What dimensions does the robot have?", "robot", "size"),
    ("Describe the phoenix's physical scale.", "phoenix", "size"),
    ("How big would the mermaid be?", "mermaid", "size"),
    ("What is the dragon's magnitude?", "dragon", "size"),
    ("Report the robot's proportions.", "robot", "size"),
    ("How much space does the phoenix occupy?", "phoenix", "size"),
    ("Characterize the mermaid's dimensions.", "mermaid", "size"),
    ("What condition currently describes the robot?", "robot", "state"),
    ("How is the dragon doing at present?", "dragon", "state"),
    ("Report the phoenix's current status.", "phoenix", "state"),
    ("What is the mermaid's present situation?", "mermaid", "state"),
    ("Is the robot operational or otherwise?", "robot", "state"),
    ("Describe the dragon's current circumstances.", "dragon", "state"),
    ("How is the phoenix faring now?", "phoenix", "state"),
    ("What condition is the mermaid presently in?", "mermaid", "state"),
)
EVALUATION_AMBIGUOUS: Tuple[str, ...] = (
    "Tell me everything relevant about the robot.",
    "Offer a brief description of the dragon.",
    "What should I know about the mermaid?",
    "Share a fact concerning the phoenix.",
    "Give an overview of the robot.",
    "Describe one notable thing about the dragon.",
    "Summarize what is known about the mermaid.",
    "Provide general information on the phoenix.",
    "What can you say about the robot?",
    "Discuss the dragon.",
    "Tell me something regarding the mermaid.",
    "Present an overview of the phoenix.",
    "Which detail matters about the robot?",
    "Recall anything about the dragon.",
    "What information concerns the mermaid?",
    "Mention the phoenix.",
)


@contextlib.contextmanager
def silent_stdout():
    """Suppress model initialization and memory-reset output."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def load_model(checkpoint: Path, device: torch.device) -> DCortexV2Model:
    """Load the real D_Cortex warm-start model for a read-only probe."""
    if not checkpoint.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if "model" not in payload or "config_model" not in payload:
        raise RuntimeError(f"Checkpoint lacks model/config_model: {checkpoint}")
    config = DCortexConfig(**payload["config_model"])
    with silent_stdout():
        model = DCortexV2Model(config)
    model.load_state_dict(payload["model"])
    del payload
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    with silent_stdout():
        model.reset_memory()
    return model


def model_buffer_hash(model: torch.nn.Module) -> str:
    """Return a byte-level SHA-256 over all model buffers."""
    digest = hashlib.sha256()
    for name, buffer in sorted(model.named_buffers(), key=lambda item: item[0]):
        raw = (
            buffer.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(buffer.shape)).encode("utf-8"))
        digest.update(str(buffer.dtype).encode("utf-8"))
        digest.update(raw)
    return digest.hexdigest()


def text_set_hash(texts: Sequence[str]) -> str:
    """Return a stable SHA-256 for a sorted text set."""
    payload = json.dumps(sorted(texts), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def gate(
    criterion_id: str,
    passed: bool,
    evidence: str,
    distribution: Dict[str, Any],
) -> Dict[str, Any]:
    """Build one frozen-gate verdict record."""
    return {
        "criterion_id": criterion_id,
        "passed": bool(passed),
        "evidence": evidence,
        "distribution": distribution,
    }


def build_likelihood_producer(
    backend: DCortexCausalLikelihoodBackend,
    attribute_margin_threshold: float,
) -> ConservativeLikelihoodQueryProducer:
    """Build a fresh conservative likelihood producer."""
    return ConservativeLikelihoodQueryProducer(
        backend=backend,
        adapter=ConservativeSemanticAdapter(),
        entity_prompt=ENTITY_PROMPT,
        attribute_prompt=ATTRIBUTE_PROMPT,
        entity_margin_threshold=0.0,
        attribute_margin_threshold=attribute_margin_threshold,
        minimum_probability=0.0,
    )


def attribute_margin(result: LikelihoodProducerResult) -> float:
    """Extract the attribute probability margin."""
    return next(score.margin for score in result.scores if score.axis == "attribute")


def calibrate_attribute_margin(backend: DCortexCausalLikelihoodBackend) -> Dict[str, Any]:
    """Choose the smallest margin yielding at least 80% calibration abstention."""
    margins: List[float] = []
    producer = build_likelihood_producer(backend, attribute_margin_threshold=0.0)
    for index, text in enumerate(CALIBRATION_AMBIGUOUS):
        result = producer.produce(
            f"cal-amb-{index}",
            index,
            text,
            ENTITY_CANDIDATES,
            ATTRIBUTE_CANDIDATES,
            provenance=(f"calibration:ambiguous:{index}",),
        )
        margins.append(attribute_margin(result))
    ordered = sorted(margins)
    target_abstentions = math.ceil(0.80 * len(ordered))
    threshold = min(1.0, ordered[target_abstentions - 1] + 0.000001)
    achieved = sum(margin < threshold for margin in margins) / len(margins)
    return {
        "threshold": threshold,
        "target_abstention": 0.80,
        "achieved_calibration_abstention": achieved,
        "ambiguous_margins": margins,
    }


def exact_attribute_label_baseline(text: str) -> str | None:
    """Return an attribute only when its canonical label occurs verbatim."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    matches = sorted(words.intersection(ATTRIBUTE_CANDIDATES))
    return matches[0] if len(matches) == 1 else None


def run(args: argparse.Namespace) -> int:
    """Run the frozen decoder-native causal-likelihood producer probe."""
    checkpoint = Path(args.checkpoint)
    results_dir = Path(args.run_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device)
    backend = DCortexCausalLikelihoodBackend(
        model=model,
        tokenizer=ENC.encode_ordinary,
        pad_token_id=EOT,
        max_seq_len=128,
        backend_version=checkpoint.name,
    )
    buffers_before = model_buffer_hash(model)
    calibration = calibrate_attribute_margin(backend)
    threshold = float(calibration["threshold"])

    evaluation_records: List[Dict[str, Any]] = []
    correct = 0
    emitted = 0
    accepted = 0
    query_only = 0
    for index, (text, expected_entity, expected_attribute) in enumerate(
        EVALUATION_QUERIES
    ):
        result = build_likelihood_producer(backend, threshold).produce(
            f"eval-{index}",
            1000 + index,
            text,
            ENTITY_CANDIDATES,
            ATTRIBUTE_CANDIDATES,
            provenance=(f"evaluation:query:{index}",),
        )
        emitted += int(result.emitted)
        accepted += int(
            result.adapter_decision is not None
            and result.adapter_decision.status == DecisionStatus.ACCEPT_QUERY
        )
        query_only += int(
            result.hypothesis is not None
            and result.hypothesis.requested_destination == RequestedDestination.QUERY_ONLY
        )
        prediction = None
        if result.hypothesis is not None:
            prediction = {
                "entity": result.hypothesis.entity_id,
                "attribute": result.hypothesis.attr_type,
            }
            correct += int(
                result.hypothesis.entity_id == expected_entity
                and result.hypothesis.attr_type == expected_attribute
            )
        evaluation_records.append(
            {
                "text": text,
                "expected_entity": expected_entity,
                "expected_attribute": expected_attribute,
                "prediction": prediction,
                "emitted": result.emitted,
                "reasons": list(result.reason_codes),
                "scores": [score.to_dict() for score in result.scores],
            }
        )
    accuracy = correct / len(EVALUATION_QUERIES)

    ambiguous_records: List[Dict[str, Any]] = []
    abstained = 0
    for index, text in enumerate(EVALUATION_AMBIGUOUS):
        result = build_likelihood_producer(backend, threshold).produce(
            f"eval-amb-{index}",
            2000 + index,
            text,
            ENTITY_CANDIDATES,
            ATTRIBUTE_CANDIDATES,
            provenance=(f"evaluation:ambiguous:{index}",),
        )
        abstained += int(not result.emitted)
        ambiguous_records.append(
            {
                "text": text,
                "emitted": result.emitted,
                "reasons": list(result.reason_codes),
                "scores": [score.to_dict() for score in result.scores],
            }
        )
    abstain_rate = abstained / len(EVALUATION_AMBIGUOUS)

    deterministic_a = build_likelihood_producer(backend, threshold).produce(
        "deterministic",
        3000,
        EVALUATION_QUERIES[0][0],
        ENTITY_CANDIDATES,
        ATTRIBUTE_CANDIDATES,
        provenance=("evaluation:deterministic",),
    ).to_dict()
    deterministic_b = build_likelihood_producer(backend, threshold).produce(
        "deterministic",
        3000,
        EVALUATION_QUERIES[0][0],
        ENTITY_CANDIDATES,
        ATTRIBUTE_CANDIDATES,
        provenance=("evaluation:deterministic",),
    ).to_dict()
    deterministic = deterministic_a == deterministic_b

    token_backend = DCortexTokenEmbeddingBackend(
        model=model,
        tokenizer=ENC.encode_ordinary,
        max_seq_len=64,
        backend_version=checkpoint.name,
    )
    token_correct = 0
    token_emitted = 0
    for index, (text, expected_entity, expected_attribute) in enumerate(
        EVALUATION_QUERIES
    ):
        baseline = ConservativePrototypeProducer(
            token_backend,
            ConservativeSemanticAdapter(),
            similarity_threshold=0.55,
            margin_threshold=0.025,
        ).produce(
            f"baseline-{index}",
            4000 + index,
            mode=HypothesisMode.QUERY,
            source_text=text,
            entity_prototypes=TOKEN_MEAN_ENTITY_PROTOTYPES,
            attr_prototypes=TOKEN_MEAN_ATTRIBUTE_PROTOTYPES,
            provenance=(f"baseline:query:{index}",),
        )
        token_emitted += int(baseline.emitted)
        token_correct += int(
            baseline.hypothesis is not None
            and baseline.hypothesis.entity_id == expected_entity
            and baseline.hypothesis.attr_type == expected_attribute
        )
    token_accuracy = token_correct / len(EVALUATION_QUERIES)
    uplift = accuracy - token_accuracy

    lexical_correct = sum(
        exact_attribute_label_baseline(text) == expected_attribute
        for text, _, expected_attribute in EVALUATION_QUERIES
    )
    lexical_accuracy = lexical_correct / len(EVALUATION_QUERIES)

    calibration_texts = [text for text, _, _ in CALIBRATION_QUERIES] + list(
        CALIBRATION_AMBIGUOUS
    )
    evaluation_texts = [text for text, _, _ in EVALUATION_QUERIES] + list(
        EVALUATION_AMBIGUOUS
    )
    overlap = sorted(set(calibration_texts).intersection(evaluation_texts))
    producer_source = (REPO_ROOT / "dcortex" / "semantic_producer.py").read_text(
        encoding="utf-8"
    )
    ast.parse(producer_source)
    forbidden = ["ALIAS_MAP", "SYNONYM_MAP", "HARDCODED_SYNONYM"]
    present = [item for item in forbidden if item in producer_source]
    buffers_after = model_buffer_hash(model)
    actual_sealed_hash = hashlib.sha256(SEALED_SOURCE.read_bytes()).hexdigest()

    gates = [
        gate(
            "Q0_ADAPTER_REQUIRED",
            accepted == emitted,
            f"Adapter accepted {accepted}/{emitted} emitted interpretations.",
            {"emitted": emitted, "accepted": accepted},
        ),
        gate(
            "Q1_NO_DIRECT_COMMIT",
            query_only == emitted,
            f"Query-only destinations {query_only}/{emitted} emitted interpretations.",
            {"emitted": emitted, "query_only": query_only},
        ),
        gate(
            "Q2_CALIBRATION_SEPARATION",
            not overlap,
            f"Exact calibration/evaluation text overlap count={len(overlap)}.",
            {
                "overlap": overlap,
                "calibration_hash": text_set_hash(calibration_texts),
                "evaluation_hash": text_set_hash(evaluation_texts),
            },
        ),
        gate(
            "Q3_DETERMINISTIC",
            deterministic,
            f"Repeated result exact={deterministic}.",
            {"exact": deterministic},
        ),
        gate(
            "Q4_NO_SYNONYM_MAP",
            not present,
            f"Forbidden lexical-map identifiers present={present}.",
            {"present": present},
        ),
        gate(
            "Q5_NOVEL_QUERY_ACCURACY",
            accuracy >= 0.75,
            f"Held-out entity+attribute accuracy={accuracy:.1%} "
            f"({correct}/{len(EVALUATION_QUERIES)}), emitted={emitted}.",
            {
                "correct": correct,
                "total": len(EVALUATION_QUERIES),
                "accuracy": accuracy,
                "emitted": emitted,
            },
        ),
        gate(
            "Q6_AMBIGUOUS_HONESTY",
            abstain_rate >= 0.80,
            f"Held-out ambiguous abstention={abstain_rate:.1%} "
            f"({abstained}/{len(EVALUATION_AMBIGUOUS)}).",
            {
                "abstained": abstained,
                "total": len(EVALUATION_AMBIGUOUS),
                "abstain_rate": abstain_rate,
            },
        ),
        gate(
            "Q7_NON_TRIVIAL_LABEL_OVERLAP",
            lexical_accuracy <= 0.10 and accuracy >= 0.75,
            f"Exact-label baseline={lexical_accuracy:.1%}; producer={accuracy:.1%}.",
            {
                "lexical_correct": lexical_correct,
                "lexical_accuracy": lexical_accuracy,
                "producer_accuracy": accuracy,
            },
        ),
        gate(
            "Q8_BASELINE_UPLIFT",
            uplift >= 0.30,
            f"Causal-likelihood accuracy={accuracy:.1%}; frozen token-mean "
            f"accuracy={token_accuracy:.1%}; uplift={uplift:+.1%}.",
            {
                "causal_likelihood_accuracy": accuracy,
                "token_mean_accuracy": token_accuracy,
                "token_mean_emitted": token_emitted,
                "uplift": uplift,
            },
        ),
        gate(
            "Q9_READ_ONLY_MODEL_STATE",
            buffers_before == buffers_after,
            f"All model buffers byte-identical={buffers_before == buffers_after}.",
            {"before": buffers_before, "after": buffers_after},
        ),
        gate(
            "Q10_SEALED_UNTOUCHED",
            actual_sealed_hash == SEALED_SHA256,
            f"Pas 7a SHA-256 actual={actual_sealed_hash}, frozen={SEALED_SHA256}.",
            {"actual": actual_sealed_hash, "frozen": SEALED_SHA256},
        ),
    ]
    all_pass = all(item["passed"] for item in gates)
    verdict = {
        "verdict": gates,
        "reference": {
            "run_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "checkpoint": str(checkpoint),
            "checkpoint_mtime": datetime.fromtimestamp(
                checkpoint.stat().st_mtime
            ).astimezone().isoformat(timespec="seconds"),
            "device": str(device),
            "backend": backend.backend_id,
            "backend_version": backend.backend_version,
            "entity_prompt": ENTITY_PROMPT,
            "attribute_prompt": ATTRIBUTE_PROMPT,
            "calibration": calibration,
            "evaluation_records": evaluation_records,
            "ambiguous_records": ambiguous_records,
            "scope": (
                "Frozen query-intent producer probe. It does not evaluate fact "
                "internalization, official F1/F3/F5, or semantic-memory advantage."
            ),
            "claim_status": "MEASURED in one local environment; not PROVEN.",
            "all_pass": all_pass,
        },
    }
    verdict_path = results_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(SEP, flush=True)
    print("[INFO] D_Cortex v15.7b causal-likelihood semantic producer probe", flush=True)
    print(f"[INFO] Checkpoint: {checkpoint}", flush=True)
    print(f"[INFO] Device: {device}", flush=True)
    print(f"[INFO] Calibrated attribute margin threshold: {threshold:.6f}", flush=True)
    print(SEP, flush=True)
    for item in gates:
        label = "✓ PASS" if item["passed"] else "✗ FAIL"
        print(f"{label} [{item['criterion_id']}] {item['evidence']}", flush=True)
    print(SEP, flush=True)
    print(f"[INFO] Overall: {'ALL GATES PASS' if all_pass else 'GATE FAILURE'}", flush=True)
    print(f"[INFO] Verdict: {verdict_path}", flush=True)
    return 0 if all_pass else 1


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    default_checkpoint = (
        REPO_ROOT.parent / "D_Cortex-main" / "runs" / "warmstart" / "warmstarted_init.pt"
    )
    parser = argparse.ArgumentParser(description="D_Cortex likelihood producer probe")
    parser.add_argument("--checkpoint", type=str, default=str(default_checkpoint))
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(REPO_ROOT / "runs" / "semantic_likelihood"),
    )
    return parser


def main() -> int:
    """CLI entry point."""
    return run(build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
