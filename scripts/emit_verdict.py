# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha local bring-up.
# Verifier: reads the artifacts produced by colab/train_local.py (run metadata,
# loss history) plus a live ENV re-check, evaluates the four pre-declared gates
# (ENV, FIT, LEARNS, RESUME) on real numbers, writes results/verdict.json and a
# loss curve PNG, and prints the verdict list. Generation and verification are
# separated: this script does not train, it only audits what is on disk.

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from verify_env_local import verify_environment  # noqa: E402

SEP: str = "=" * 70


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def gate_env() -> Dict[str, Any]:
    result = verify_environment()
    return {"criterion_id": "ENV", "passed": bool(result["passed"]),
            "evidence": result["evidence"]}


def gate_fit(run_meta: Optional[Dict[str, Any]], vram_limit_gb: float) -> Dict[str, Any]:
    if run_meta is None:
        return {"criterion_id": "FIT", "passed": False,
                "evidence": "main run metadata missing."}
    peak = run_meta.get("peak_vram_alloc_gb")
    peak_reserved = run_meta.get("peak_vram_reserved_gb")
    peak_at_50 = run_meta.get("peak_vram_at_step50_gb")
    passed = peak is not None and peak < vram_limit_gb
    evidence = (f"peak alloc {peak} GB (< {vram_limit_gb} GB target); "
                f"peak at step50 {peak_at_50} GB; peak reserved {peak_reserved} GB; "
                f"steps {run_meta.get('start_step')}->{run_meta.get('end_step')}; "
                f"oom_skips {run_meta.get('oom_steps')}")
    return {"criterion_id": "FIT", "passed": bool(passed), "evidence": evidence}


def gate_learns(loss_history: Optional[Dict[str, Any]], drop_target: float,
                window: int) -> Tuple[Dict[str, Any], List[Tuple[int, float]]]:
    if loss_history is None or not loss_history.get("loss_history"):
        return ({"criterion_id": "LEARNS", "passed": False,
                 "evidence": "loss history missing."}, [])
    history: List[Tuple[int, float]] = [(int(s), float(v))
                                        for s, v in loss_history["loss_history"]]
    losses = [v for _s, v in history]
    n = len(losses)
    w = max(1, min(window, n // 3))
    initial_smoothed = statistics.mean(losses[:w])
    final_smoothed = statistics.mean(losses[-w:])
    drop_pct = (initial_smoothed - final_smoothed) / initial_smoothed * 100.0
    passed = drop_pct >= drop_target * 100.0
    evidence = (
        f"steps={n}; drop method=mean(first {w}) vs mean(last {w}); "
        f"initial_smoothed={initial_smoothed:.4f}; final_smoothed={final_smoothed:.4f}; "
        f"drop={drop_pct:.1f}% (target >= {drop_target * 100:.0f}%); "
        f"raw_first={losses[0]:.4f}; raw_last={losses[-1]:.4f}; "
        f"min={min(losses):.4f}; median={statistics.median(losses):.4f}; "
        f"max={max(losses):.4f}"
    )
    return ({"criterion_id": "LEARNS", "passed": bool(passed),
             "evidence": evidence}, history)


def gate_resume(p1: Optional[Dict[str, Any]],
                p2: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if p1 is None or p2 is None:
        return {"criterion_id": "RESUME", "passed": False,
                "evidence": "resume metadata missing (need run_meta_resume_p1/p2)."}
    p1_end = p1.get("end_step")
    p2_start = p2.get("start_step")
    p2_resumed = bool(p2.get("resumed"))
    p2_first = p2.get("resumed_first_loss")
    p1_first = p1.get("loss_first")
    p1_last = p1.get("loss_last")
    step_continuous = (p2_start == p1_end and p2_start is not None and p2_start > 0)
    # Continuous loss: the first post-resume loss must stay on the trajectory
    # (well below the fresh initial loss), not reset to a cold-start value.
    loss_continuous = (p2_first is not None and p1_first is not None
                       and p2_first < p1_first)
    passed = p2_resumed and step_continuous and loss_continuous
    evidence = (
        f"p1 ran ->{p1_end} (first={p1_first:.4f}, last={p1_last:.4f}); "
        f"p2 resumed={p2_resumed} from start_step={p2_start}; "
        f"step_continuous={step_continuous}; "
        f"first_loss_after_resume={p2_first:.4f} "
        f"(< p1 cold-start {p1_first:.4f} -> on trajectory, no reset); "
        f"p2 end_step={p2.get('end_step')}, n_logged={p2.get('n_logged')}"
    )
    return {"criterion_id": "RESUME", "passed": bool(passed), "evidence": evidence}


def plot_loss_curve(history: List[Tuple[int, float]], out_path: Path) -> None:
    if not history:
        return
    steps = [s for s, _v in history]
    losses = [v for _s, v in history]
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, linewidth=1.0, color="#1f77b4", label="train_loss (aggregate)")
    plt.xlabel("step")
    plt.ylabel("train loss (structural total + LM, accum mean)")
    plt.title("D_Cortex v2.0-alpha local training loss (RTX 5080, bf16)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[INFO] Loss curve saved: {out_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the four bring-up gates")
    repo_root = SCRIPTS_DIR.parent
    parser.add_argument("--run-dir", type=str,
                        default=str(repo_root / "runs" / "dcortex"))
    parser.add_argument("--resume-dir", type=str,
                        default=str(repo_root / "runs" / "dcortex_resume"))
    parser.add_argument("--vram-limit", type=float, default=15.0)
    parser.add_argument("--learn-drop", type=float, default=0.15)
    parser.add_argument("--learn-window", type=int, default=10)
    args = parser.parse_args()

    run_results = Path(args.run_dir) / "results"
    resume_results = Path(args.resume_dir) / "results"

    run_meta = _load_json(run_results / "run_meta_main.json")
    loss_history = _load_json(run_results / "loss_history.json")
    p1 = _load_json(resume_results / "run_meta_resume_p1.json")
    p2 = _load_json(resume_results / "run_meta_resume_p2.json")

    print(SEP, flush=True)
    print("[INFO] Evaluating bring-up gates", flush=True)
    print(SEP, flush=True)

    env = gate_env()
    fit = gate_fit(run_meta, args.vram_limit)
    learns, history = gate_learns(loss_history, args.learn_drop, args.learn_window)
    resume = gate_resume(p1, p2)
    verdict: List[Dict[str, Any]] = [env, fit, learns, resume]

    plot_loss_curve(history, run_results / "loss_curve.png")

    verdict_path = run_results / "verdict.json"
    with open(verdict_path, "w", encoding="utf-8") as handle:
        json.dump(verdict, handle, indent=2)

    print(SEP, flush=True)
    for item in verdict:
        tag = "✓ PASS" if item["passed"] else "✗ FAIL"
        print(f"{tag}  [{item['criterion_id']}] {item['evidence']}", flush=True)
    print(SEP, flush=True)
    all_pass = all(item["passed"] for item in verdict)
    print(f"[INFO] Overall: {'ALL GATES PASS' if all_pass else 'ONE OR MORE GATES FAILED'}",
          flush=True)
    print("VERDICT_JSON " + json.dumps(verdict), flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
