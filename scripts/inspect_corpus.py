# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha local training campaign.
# PHASE 1 inspector: enumerate a JSONL corpus folder, detect schema/encoding/
# language, estimate token counts with the repo tokenizer (tiktoken gpt2), and
# classify the training route (DUAL_AGENT vs LM_DECODER). Sampling-based: it
# reads a bounded prefix of each file and extrapolates by bytes, so it stays
# fast on multi-GB files. Read-only, writes nothing except a JSON summary.

import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

import tiktoken

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

SEP: str = "=" * 70
DATA_FOLDER: str = r"E:\DATA\training_data_clean"
SAMPLE_BYTES: int = 6_000_000          # prefix read per file for estimation
SAMPLE_MAX_RECORDS: int = 20_000       # cap records parsed per file

ENC = tiktoken.get_encoding("gpt2")

# Route-signalling key sets.
_FQA_KEYS = ({"fact", "question", "answer"}, {"context", "query"},
             {"question", "answer"}, {"prompt", "completion"},
             {"instruction", "output"})

# Lightweight language heuristic markers.
_RO_CHARS = set("ăâîșțşţ")
_RO_WORDS = {"și", "este", "pentru", "care", "sunt", "într", "această",
             "cu", "să", "nu", "din", "the"}  # 'the' excluded below
_RO_WORDS.discard("the")
_RO_STOP = {"și", "este", "pentru", "care", "sunt", "din", "să", "nu", "cu",
            "într", "acest", "această", "mai", "fără", "după"}
_EN_STOP = {"the", "and", "is", "of", "to", "in", "that", "for", "with",
            "this", "are", "was", "you", "it"}


def detect_encoding(raw: bytes) -> str:
    """Return a short encoding label for a byte prefix."""
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig (BOM)"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "non-utf-8 (latin-1 fallback)"


def language_mix(text: str) -> Dict[str, float]:
    """Heuristic Romanian vs English share from stopword and diacritic hits."""
    lowered = text.lower()
    ro_diacritics = sum(lowered.count(c) for c in _RO_CHARS)
    tokens = lowered.split()
    if not tokens:
        return {"ro_score": 0.0, "en_score": 0.0, "ro_diacritics_per_kchar": 0.0}
    ro_hits = sum(1 for t in tokens if t.strip(".,!?;:()[]\"'") in _RO_STOP)
    en_hits = sum(1 for t in tokens if t.strip(".,!?;:()[]\"'") in _EN_STOP)
    total = max(1, ro_hits + en_hits)
    return {
        "ro_score": round(100.0 * ro_hits / total, 1),
        "en_score": round(100.0 * en_hits / total, 1),
        "ro_diacritics_per_kchar": round(1000.0 * ro_diacritics / max(1, len(lowered)), 2),
    }


def inspect_file(path: str) -> Dict[str, Any]:
    """Sample a JSONL file and return schema, encoding, language, token stats."""
    size_bytes = os.path.getsize(path)
    with open(path, "rb") as handle:
        raw_prefix = handle.read(SAMPLE_BYTES)
    encoding = detect_encoding(raw_prefix)
    text_prefix = raw_prefix.decode("utf-8", errors="replace")

    key_sets: Counter = Counter()
    text_field_counts: Counter = Counter()
    parse_errors = 0
    records = 0
    sample_text_parts: List[str] = []
    sample_first_snippet = ""

    # Drop a possibly-truncated last line.
    lines = text_prefix.split("\n")
    if len(lines) > 1:
        lines = lines[:-1]

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if records >= SAMPLE_MAX_RECORDS:
            break
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
            continue
        if not isinstance(obj, dict):
            text_field_counts["<non-dict>"] += 1
            records += 1
            continue
        key_sets[frozenset(obj.keys())] += 1
        text_val = ""
        for field in ("text", "content", "story", "body"):
            if isinstance(obj.get(field), str) and obj.get(field):
                text_val = obj[field]
                text_field_counts[field] += 1
                break
        if not text_val:
            for k, v in obj.items():
                if isinstance(v, str) and v:
                    text_val = v
                    text_field_counts[f"<first-str:{k}>"] += 1
                    break
        if text_val:
            if not sample_first_snippet:
                sample_first_snippet = text_val[:240].replace("\n", " ")
            if sum(len(p) for p in sample_text_parts) < 2_000_000:
                sample_text_parts.append(text_val)
        records += 1

    sample_text = "\n".join(sample_text_parts)
    sample_tokens = len(ENC.encode_ordinary(sample_text)) if sample_text else 0
    # tokens per total prefix byte (json overhead included) for a robust extrapolation
    bytes_consumed = len(raw_prefix)
    tokens_per_byte = sample_tokens / max(1, bytes_consumed)
    est_total_tokens = int(tokens_per_byte * size_bytes)
    avg_tokens_per_record = sample_tokens / max(1, records)
    est_records = int(size_bytes / max(1, bytes_consumed) * records)

    top_schema = key_sets.most_common(3)
    has_fqa = any(any(fq.issubset(ks) for fq in _FQA_KEYS) for ks in key_sets)

    return {
        "file": os.path.basename(path),
        "size_bytes": size_bytes,
        "size_gb": round(size_bytes / (1024 ** 3), 3),
        "encoding": encoding,
        "parse_errors_in_sample": parse_errors,
        "sample_records": records,
        "top_schemas": [(sorted(list(ks)), n) for ks, n in top_schema],
        "text_fields": dict(text_field_counts),
        "has_fact_question_answer_keys": has_fqa,
        "sample_tokens": sample_tokens,
        "tokens_per_byte": round(tokens_per_byte, 4),
        "est_total_tokens": est_total_tokens,
        "est_records": est_records,
        "avg_tokens_per_record": round(avg_tokens_per_record, 1),
        "language": language_mix(sample_text[:1_500_000]),
        "first_snippet": sample_first_snippet,
    }


def main() -> int:
    if not os.path.isdir(DATA_FOLDER):
        print(f"[ERROR] DATA_FOLDER not found: {DATA_FOLDER}", flush=True)
        return 1
    files = sorted(f for f in os.listdir(DATA_FOLDER)
                   if f.lower().endswith((".jsonl", ".json", ".txt")))
    print(SEP, flush=True)
    print(f"[INFO] Inspecting {len(files)} files in {DATA_FOLDER}", flush=True)
    print(f"[INFO] Tokenizer: tiktoken gpt2 (vocab {ENC.n_vocab}); "
          f"sample {SAMPLE_BYTES // 1_000_000} MB/file", flush=True)
    print(SEP, flush=True)

    results: List[Dict[str, Any]] = []
    for name in files:
        path = os.path.join(DATA_FOLDER, name)
        print(f"[INFO] -> {name} ...", flush=True)
        info = inspect_file(path)
        results.append(info)
        lang = info["language"]
        print(f"    size={info['size_gb']} GB  enc={info['encoding']}  "
              f"est_tokens={info['est_total_tokens']:,}  "
              f"avg_tok/rec={info['avg_tokens_per_record']}  "
              f"RO={lang['ro_score']}% EN={lang['en_score']}% "
              f"(diacritics/kchar={lang['ro_diacritics_per_kchar']})", flush=True)
        print(f"    schema={info['top_schemas']}  fqa_keys={info['has_fact_question_answer_keys']}",
              flush=True)
        print(f"    snippet: {info['first_snippet'][:160]}", flush=True)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "corpus_inspection.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    total_tokens = sum(r["est_total_tokens"] for r in results)
    print(SEP, flush=True)
    print(f"[INFO] Aggregate est tokens (ALL files, includes pipeline-stage "
          f"duplication): {total_tokens:,}", flush=True)
    print(f"[INFO] Wrote {out_path}", flush=True)
    print(SEP, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
