# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha -- episode adapter: convert instructional records with
# [SYSTEM]/[USER]/[ASSISTANT] turns into v11 DUAL_AGENT episodes whose facts are
# extracted from the declarative ([ASSISTANT]) content via relation-cloze. Each
# fact is keyed by its subject; the answer token lives ONLY in the fact written
# to memory, never in the query, so the decoder must retrieve the correct fact
# from memory to answer. The dcortex/ architecture is NOT modified.

import argparse
import json
import os
import re
import sys
import zlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

import tiktoken

SEP: str = "=" * 70
SPLIT_SEED: int = 42
DEFAULT_SOURCE: str = r"E:\DATA\training_data_clean\06_FINAL_TRAINING_READY-003_BYON.jsonl"

ENC = tiktoken.get_encoding("gpt2")
EOT: int = ENC.eot_token

# Exact turn markers only (uppercase). [CODE], [1], [json] etc. are NOT markers.
MARKER_RE = re.compile(r"\[(SYSTEM|USER|ASSISTANT)\]")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
LINK_VERBS = {"is", "are", "was", "were", "be", "been", "has", "have", "had",
              "became", "become", "contains", "includes", "consists", "refers",
              "describes", "represents", "remains", "comprises", "denotes"}
_PUNCT = ".,;:!?()[]{}\"'`"
_NUM_RE = re.compile(r"^\d[\d,\.]*$")


@dataclass
class FactInfo:
    text: str
    answer_token_id: int
    subject: str
    value: str
    query: str


@dataclass
class Episode:
    facts: List[Dict]            # serialized FactInfo dicts
    prompt: str
    target_fact_idx: int
    answer_token_id: int
    ep_type: str


def is_held_out(record_index: int) -> bool:
    h = zlib.crc32(f"{SPLIT_SEED}:{record_index}".encode("utf-8"))
    return (h % 10) == 0  # ~10% records held out


def parse_turns(text: str) -> Tuple[Optional[List[Tuple[str, str]]], str]:
    """Parse [SYSTEM]/[USER]/[ASSISTANT] turns. Returns (turns, skip_reason)."""
    markers = list(MARKER_RE.finditer(text))
    if not markers:
        return None, "no_markers"
    turns: List[Tuple[str, str]] = []
    for i, m in enumerate(markers):
        role = m.group(1)
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        content = text[start:end].strip()
        if content:
            turns.append((role, content))
    if not turns:
        return None, "empty_turns"
    if not any(r == "ASSISTANT" for r, _ in turns):
        return None, "no_assistant"
    return turns, ""


def _salience(word: str, sentence_start: bool) -> int:
    w = word.strip(_PUNCT)
    if not w:
        return 0
    if _NUM_RE.match(w):
        return 3                                   # numbers: most distinctive
    if (not sentence_start) and w[0].isupper() and len(w) >= 3:
        return 2                                   # proper noun (not sentence start)
    if len(w) >= 8 and w.isalpha():
        return 1                                   # long content word
    return 0


def extract_cloze_facts(assistant_text: str) -> List[FactInfo]:
    """Extract (subject, value) cloze facts from declarative content. The query
    is the clause truncated before the value; the answer is the value's first
    gpt2 token; the value is verified absent from the query."""
    facts: List[FactInfo] = []
    sentences = SENT_SPLIT_RE.split(assistant_text)
    for sent in sentences:
        sent = sent.strip()
        words = sent.split()
        if len(words) < 6 or len(words) > 60:
            continue
        # Subject end = first linking verb within the first 8 words, else 4.
        subj_end = None
        for i, w in enumerate(words[:8]):
            if w.strip(_PUNCT).lower() in LINK_VERBS:
                subj_end = i
                break
        if subj_end is None:
            subj_end = min(4, len(words) // 2)
        if subj_end < 1:
            continue
        subject = " ".join(words[:subj_end]).strip()
        subject_lower = subject.lower()
        # Subject must carry a real content word (skip generic "Most", "It is").
        if not any(len(w.strip(_PUNCT)) >= 4 for w in words[:subj_end]):
            continue
        # Best value: highest salience after the subject, earliest on ties,
        # not part of the subject. Require salience >= 2 (proper noun or number)
        # so the answer is NOT guessable from language priors -> memory matters.
        best_j = None
        best_score = 0
        for j in range(subj_end + 1, len(words)):
            raw = words[j]
            w = raw.strip(_PUNCT)
            if not w or w.lower() in subject_lower:
                continue
            score = _salience(raw, sentence_start=False)
            if score > best_score:
                best_score = score
                best_j = j
        if best_j is None or best_score < 2:
            continue
        value = words[best_j].strip(_PUNCT)
        query = " ".join(words[:best_j]).strip()
        if not query or value.lower() in query.lower():
            continue
        # Single-token answer = first gpt2 token of " value".
        ans_ids = ENC.encode_ordinary(" " + value)
        if not ans_ids:
            continue
        answer_token_id = ans_ids[0]
        # Query must fit and retain the subject (keep within seq_len budget).
        query_ids = ENC.encode_ordinary(query)
        if len(query_ids) > 60:
            query_ids = query_ids[:60]
            query = ENC.decode(query_ids)
        if subject_lower not in query.lower():
            continue
        # Token-level leakage guard: the answer token must NOT appear anywhere in
        # the query token sequence, so the decoder cannot copy it without memory.
        if answer_token_id in query_ids:
            continue
        facts.append(FactInfo(text=sent, answer_token_id=answer_token_id,
                              subject=subject, value=value, query=query))
    return facts


def bundle_episodes(facts: List[FactInfo], system_fact: Optional[FactInfo],
                    episode_facts: int, rng_seed: int) -> List[Episode]:
    """Bundle K distinct-subject facts; emit one episode per queried fact."""
    import random
    rng = random.Random(rng_seed)
    pool = facts[:]
    rng.shuffle(pool)
    episodes: List[Episode] = []
    i = 0
    while i < len(pool):
        bundle: List[FactInfo] = []
        seen_subjects = set()
        while i < len(pool) and len(bundle) < episode_facts:
            f = pool[i]
            i += 1
            key = f.subject.lower()
            if key in seen_subjects:
                continue
            seen_subjects.add(key)
            bundle.append(f)
        if len(bundle) < 2:
            continue
        written = list(bundle)
        if system_fact is not None:
            written = [system_fact] + written        # SYSTEM as an extra non-target fact
        offset = 1 if system_fact is not None else 0
        for q_idx, f in enumerate(bundle):
            episodes.append(Episode(
                facts=[asdict(w) for w in written],
                prompt=f.query,
                target_fact_idx=q_idx + offset,
                answer_token_id=f.answer_token_id,
                ep_type="cloze",
            ))
    return episodes


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    print(f"[INFO] Episode adapter over {os.path.basename(args.source)}", flush=True)
    print(f"[INFO] EPISODE_FACTS={args.episode_facts} INCLUDE_SYSTEM_AS_FACT="
          f"{args.include_system} max_records={args.max_records}", flush=True)
    print(SEP, flush=True)

    if not os.path.exists(args.source):
        raise RuntimeError(f"Source not found: {args.source}. No synthetic fallback.")

    counts = {"records": 0, "marker_bearing": 0, "parsed": 0,
              "skip_no_markers": 0, "skip_empty_turns": 0, "skip_no_assistant": 0,
              "skip_json_error": 0, "skip_no_text": 0, "yielded_facts": 0,
              "records_zero_facts": 0}
    facts_per_record: List[int] = []
    train_facts: List[FactInfo] = []
    heldout_facts: List[FactInfo] = []
    train_system: Optional[FactInfo] = None
    heldout_system: Optional[FactInfo] = None
    record_index = 0

    with open(args.source, "r", encoding="utf-8") as handle:
        for line in handle:
            if counts["records"] >= args.max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                counts["skip_json_error"] += 1
                continue
            text = obj.get("text", "")
            if not text:
                counts["skip_no_text"] += 1
                continue
            counts["records"] += 1
            if MARKER_RE.search(text):
                counts["marker_bearing"] += 1
            turns, reason = parse_turns(text)
            if turns is None:
                counts[f"skip_{reason}"] = counts.get(f"skip_{reason}", 0) + 1
                continue
            held = is_held_out(record_index)
            record_index += 1
            assistant = " ".join(c for r, c in turns if r == "ASSISTANT")
            rec_facts = extract_cloze_facts(assistant)
            facts_per_record.append(len(rec_facts))
            if not rec_facts:
                counts["records_zero_facts"] += 1
                continue
            counts["parsed"] += 1
            counts["yielded_facts"] += len(rec_facts)
            if args.include_system and (train_system is None or heldout_system is None):
                sys_text = next((c for r, c in turns if r == "SYSTEM"), None)
                if sys_text:
                    sys_fact = FactInfo(text=sys_text[:300], answer_token_id=EOT,
                                        subject="[system]", value="", query="")
                    if held and heldout_system is None:
                        heldout_system = sys_fact
                    elif (not held) and train_system is None:
                        train_system = sys_fact
            (heldout_facts if held else train_facts).extend(rec_facts)

    sys_train = train_system if args.include_system else None
    sys_held = heldout_system if args.include_system else None
    train_eps = bundle_episodes(train_facts, sys_train, args.episode_facts, SPLIT_SEED)
    held_eps = bundle_episodes(heldout_facts, sys_held, args.episode_facts, SPLIT_SEED + 1)

    for name, eps in [("train", train_eps), ("heldout", held_eps)]:
        path = out_dir / f"episodes_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for ep in eps:
                handle.write(json.dumps(asdict(ep), ensure_ascii=False) + "\n")
        print(f"[INFO] Wrote {len(eps):,} {name} episodes -> {path.name}", flush=True)

    nonzero = [c for c in facts_per_record if c > 0]
    nonzero.sort()
    def _q(p: float) -> int:
        return nonzero[min(len(nonzero) - 1, int(p * len(nonzero)))] if nonzero else 0
    parse_rate = counts["parsed"] / max(1, counts["marker_bearing"])

    stats = {
        "source": os.path.basename(args.source),
        "episode_facts": args.episode_facts,
        "include_system_as_fact": args.include_system,
        "counts": counts,
        "parse_rate_of_marker_bearing": round(parse_rate, 4),
        "facts_per_record_min": (min(nonzero) if nonzero else 0),
        "facts_per_record_median": (_q(0.5)),
        "facts_per_record_max": (max(nonzero) if nonzero else 0),
        "train_facts": len(train_facts),
        "heldout_facts": len(heldout_facts),
        "train_episodes": len(train_eps),
        "heldout_episodes": len(held_eps),
    }
    with open(out_dir / "adapter_stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(SEP, flush=True)
    print(f"[INFO] records={counts['records']:,} marker_bearing={counts['marker_bearing']:,} "
          f"parsed(>=1 fact)={counts['parsed']:,} ({parse_rate:.1%} of marker-bearing)",
          flush=True)
    print(f"[INFO] skips: no_assistant={counts.get('skip_no_assistant',0)} "
          f"empty_turns={counts.get('skip_empty_turns',0)} "
          f"json_error={counts['skip_json_error']} zero_facts={counts['records_zero_facts']}",
          flush=True)
    print(f"[INFO] facts/record (records with >=1): min={stats['facts_per_record_min']} "
          f"median={stats['facts_per_record_median']} max={stats['facts_per_record_max']}",
          flush=True)
    print(f"[INFO] train: {len(train_facts):,} facts -> {len(train_eps):,} episodes | "
          f"heldout: {len(heldout_facts):,} facts -> {len(held_eps):,} episodes", flush=True)
    print(SEP, flush=True)
    print("ADAPTER_STATS_JSON " + json.dumps(stats), flush=True)
    return 0


def build_argparser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="D_Cortex episode adapter (relation cloze)")
    p.add_argument("--source", type=str, default=DEFAULT_SOURCE)
    p.add_argument("--out-dir", type=str, default=str(repo_root / "runs" / "adapter"))
    p.add_argument("--episode-facts", type=int, default=6)
    p.add_argument("--include-system", action="store_true", default=True)
    p.add_argument("--no-system", dest="include_system", action="store_false")
    p.add_argument("--max-records", type=int, default=80000)
    return p


def main() -> int:
    return run(build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
