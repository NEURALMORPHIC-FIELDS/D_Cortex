# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Builds the D_Cortex_IPCAnalyst real-scale professional pack from the official
# International Patent Classification (IPC, WIPO) class titles. The source is pinned
# by GitHub commit + file SHA and payload-validated; every committed fact is a real
# IPC code -> official title (cleaned deterministically of scraped HTML/version
# markers). Codes that cannot be cleaned to a valid title do NOT enter committed.
# Data construction only; no model is loaded here.

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

SEP = "=" * 70
REPO_ROOT = Path(__file__).resolve().parent.parent
PACK_DIR = REPO_ROOT / "data" / "professional_ipc" / "D_Cortex_IPCAnalyst"
SOURCE_DIR = REPO_ROOT / "data" / "professional_ipc" / "source"

# pinned source: WIPO IPC class titles (English), via a fixed GitHub commit
IPC_REPO = "thxare/IPCcodes"
IPC_PATH = "funcionesInlges/section.json"
IPC_COMMIT = "0ff722fbaea591b329af6f33d74fdbb7b6b2246a"
IPC_URL = f"https://raw.githubusercontent.com/{IPC_REPO}/{IPC_COMMIT}/{IPC_PATH}"

CODE_RE = re.compile(r"^[A-H]\d{2}[A-Z]?$")


def clean_title(desc: str) -> str:
    d = str(desc).split('"')[0]                          # cut at first stray HTML quote
    d = re.split(r"\s+\d{4}\.\d{2}", d)[0]               # cut at IPC version date (e.g. 2006.01)
    d = re.sub(r"<[^>]+>", "", d)                        # strip any HTML tags
    d = re.sub(r"\s*\[\d+\]\s*$", "", d)                 # strip trailing reform marker [2]
    return d.strip().rstrip(";").strip()


def fetch_validate_pin() -> List[Tuple[str, str, str]]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(
        urllib.request.Request(IPC_URL, headers={"User-Agent": "DCortex/1.0"}), timeout=30).read()
    if len(raw) <= 5120:
        raise RuntimeError(f"IPC payload {len(raw)} bytes <= 5 KB (rejected)")
    if raw.lstrip()[:1] not in (b"[", b"{"):
        raise RuntimeError("IPC payload does not start as JSON (redirect/error body)")
    data = json.loads(raw)
    if not isinstance(data, list) or len(data) < 100:
        raise RuntimeError(f"IPC payload not a list of >= 100 entries (got {type(data)} len {len(data) if isinstance(data, list) else 'n/a'})")
    facts: List[Tuple[str, str, str]] = []
    seen = set()
    for e in data:
        code = str(e.get("code", "")).strip()
        title = clean_title(e.get("description", ""))
        category = str(e.get("category", "")).strip()
        if not CODE_RE.match(code) or not (5 <= len(title) <= 200):
            continue
        if code in seen:
            continue
        seen.add(code)
        facts.append((code, title, category))
    if len(facts) < 100:
        raise RuntimeError(f"only {len(facts)} clean IPC facts (< 100); refusing to fabricate")
    sha = hashlib.sha256(raw).hexdigest()
    pinned = SOURCE_DIR / f"ipc_titles_{IPC_COMMIT[:8]}_{sha[:16]}.json"
    pinned.write_bytes(raw)
    print(f"[INFO] IPC: {len(facts)} clean class titles; pinned {pinned.name} sha {sha[:16]}", flush=True)
    return facts


def provenance(code: str) -> Dict[str, str]:
    return {"source": "wipo_ipc", "ref": f"IPC class {code}",
            "document": f"WIPO International Patent Classification, class {code}",
            "pin": f"github {IPC_REPO}@{IPC_COMMIT[:8]} {IPC_PATH}"}


# small, honest illustrative non-committed regions (not fabricated facts; meta rules)
FORBIDDEN = [
    {"pattern": "is patentable", "reason": "patentability is a legal determination, not an IPC fact",
     "severity": "high"},
    {"pattern": "guaranteed to be granted", "reason": "grant is an examination outcome, not in this pack",
     "severity": "high"},
    {"pattern": "infringes", "reason": "infringement is a legal conclusion outside this classification pack",
     "severity": "high"},
]
DISPUTED: List[Dict[str, Any]] = []   # IPC titles are authoritative; no disputed entries
PROVISIONAL: List[Dict[str, Any]] = []


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> str:
    body = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n"
    enc = body.encode("utf-8")
    path.write_bytes(enc)
    return hashlib.sha256(enc).hexdigest()


def write_json(path: Path, obj: Any) -> str:
    enc = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    path.write_bytes(enc)
    return hashlib.sha256(enc).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the D_Cortex_IPCAnalyst real-scale pack")
    ap.add_argument("--out", default=str(PACK_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    print("[INFO] Building D_Cortex_IPCAnalyst (official WIPO IPC class titles)", flush=True)

    facts = fetch_validate_pin()
    committed_rows: List[Dict[str, Any]] = []
    categories = {}
    for code, title, category in facts:
        committed_rows.append({"entity": code, "attribute": "title", "value": title,
                               "provenance": provenance(code)})
        if category:
            categories.setdefault(code, category)
    # second attribute: category (also real, from the same pinned source)
    for code, _t, category in facts:
        if category:
            committed_rows.append({"entity": code, "attribute": "category", "value": category,
                                   "provenance": provenance(code)})

    schemas = {
        "domain": "patent_classification_ipc",
        "entities": [c for c, _t, _cat in facts],
        "attributes": {"title": "string", "category": "string"},
        "in_domain_keywords": ["ipc", "patent", "classification", "class", "code", "section",
                               "cover", "covers", "wipo", "title", "category", "subclass"],
    }
    abstain_rules = {
        "rules": [
            {"id": "R1_unknown", "when": "code not in committed", "action": "abstain",
             "message": "Not grounded in the IPC committed memory."},
            {"id": "R2_out_of_domain", "when": "no in-domain keyword and no known code", "action": "out_of_domain",
             "message": "Query is outside the IPC classification domain."},
            {"id": "R4_forbidden", "when": "answer matches a forbidden pattern", "action": "block",
             "message": "Legal conclusion is forbidden in this classification pack."},
        ],
        "default_action": "abstain",
    }
    sources = {"wipo_ipc": {"type": "official_taxonomy_pinned",
                            "ref": f"github {IPC_REPO}@{IPC_COMMIT} {IPC_PATH}",
                            "note": "WIPO International Patent Classification class titles (English), "
                                    "pinned by commit + file SHA, HTML/version markers cleaned"}}

    shas = {
        "committed.jsonl": write_jsonl(out / "committed.jsonl", committed_rows),
        "provisional.jsonl": write_jsonl(out / "provisional.jsonl", PROVISIONAL),
        "disputed.jsonl": write_jsonl(out / "disputed.jsonl", DISPUTED),
        "forbidden.jsonl": write_jsonl(out / "forbidden.jsonl", FORBIDDEN),
        "sources.json": write_json(out / "sources.json", sources),
        "abstain_rules.json": write_json(out / "abstain_rules.json", abstain_rules),
        "schemas.json": write_json(out / "schemas.json", schemas),
    }
    write_json(out / "pack_manifest.json",
               {"pack": "D_Cortex_IPCAnalyst", "files": shas,
                "counts": {"committed": len(committed_rows), "ipc_codes": len(facts),
                           "forbidden": len(FORBIDDEN)},
                "source_commit": IPC_COMMIT})
    for name, sha in shas.items():
        print(f"  ✓ {name:20s} sha {sha[:16]}", flush=True)
    print(f"[INFO] committed={len(committed_rows)} (codes={len(facts)}, title+category) forbidden={len(FORBIDDEN)}",
          flush=True)
    print(SEP, flush=True)
    print("IPC_PACK_BUILT D_Cortex_IPCAnalyst", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
