# -- coding: utf-8 --
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b-RB4 validated independent source acquisition and pinning.

import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
SEP = "=" * 70

REST_ALL_URL = "https://restcountries.com/v3.1/all?fields=name,capital"
REST_INDEPENDENT_URL = (
    "https://restcountries.com/v3.1/independent?status=true&fields=name,capital"
)
FALLBACK_REPO = "samayo/country-json"
FALLBACK_COMMIT = "41d4084bc1ccf9614dab45255a41ba3a5473be74"
FALLBACK_PATH = "src/country-by-capital-city.json"
FALLBACK_URL = (
    "https://raw.githubusercontent.com/"
    f"{FALLBACK_REPO}/{FALLBACK_COMMIT}/{FALLBACK_PATH}"
)
FALLBACK_LICENSE_URL = (
    "https://raw.githubusercontent.com/"
    f"{FALLBACK_REPO}/{FALLBACK_COMMIT}/LICENSE"
)

SOURCE_DIR = REPO_ROOT / "data" / "rb4" / "source"
MANIFEST_PATH = SOURCE_DIR / "pinned_source_manifest.json"
FAILED_FETCH_PATH = SOURCE_DIR / "failed_fetch_report.json"

MIN_RESPONSE_BYTES = 5 * 1024
MIN_SOURCE_RECORDS = 180


@dataclass(frozen=True)
class HttpFetchResult:
    """One fetched source payload with first-class HTTP evidence."""

    name: str
    url: str
    status: int | None
    body: bytes
    error: str | None


def sha256_bytes(payload: bytes) -> str:
    """Return SHA-256 for bytes."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Return SHA-256 for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_url(name: str, url: str) -> HttpFetchResult:
    """Fetch one URL without saving its body."""
    request = urllib.request.Request(url, headers={"User-Agent": "D_Cortex-RB4"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return HttpFetchResult(
                name=name,
                url=url,
                status=int(response.status),
                body=response.read(),
                error=None,
            )
    except urllib.error.HTTPError as exc:
        return HttpFetchResult(
            name=name,
            url=url,
            status=int(exc.code),
            body=exc.read(),
            error=str(exc),
        )
    except urllib.error.URLError as exc:
        return HttpFetchResult(name=name, url=url, status=None, body=b"", error=str(exc))


def first_200(payload: bytes) -> str:
    """Return a printable first-200-byte response excerpt."""
    return payload[:200].decode("utf-8", errors="replace").replace("\r", "\\r")


def country_capital_from_entry(entry: Mapping[str, Any]) -> Tuple[str, str]:
    """Extract one country-capital fact from REST Countries or the pinned fallback."""
    if "name" in entry and "capital" in entry:
        name = entry["name"]
        country = (
            str(name.get("common", "")).strip()
            if isinstance(name, Mapping)
            else str(name).strip()
        )
        capital_data = entry["capital"]
        capital = (
            str(capital_data[0]).strip()
            if isinstance(capital_data, list) and capital_data
            else str(capital_data).strip()
        )
        return country, capital
    if "country" in entry and "city" in entry:
        return str(entry["country"]).strip(), str(entry["city"]).strip()
    if "country" in entry and "capital" in entry:
        return str(entry["country"]).strip(), str(entry["capital"]).strip()
    return "", ""


def validate_payload(result: HttpFetchResult) -> Tuple[bool, List[Dict[str, str]], str]:
    """Validate mandatory source constraints and normalize only factual records."""
    if result.status != 200:
        return False, [], f"HTTP status {result.status}, expected 200"
    if len(result.body) <= MIN_RESPONSE_BYTES:
        return False, [], f"body size {len(result.body)} <= {MIN_RESPONSE_BYTES}"
    try:
        payload = json.loads(result.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, [], f"JSON parse failed: {exc}"
    if isinstance(payload, Mapping):
        if payload.get("success") is False or "errors" in payload:
            return False, [], "top-level error payload"
        return False, [], "JSON payload is an object, expected list"
    if not isinstance(payload, list):
        return False, [], f"JSON payload is {type(payload).__name__}, expected list"
    if len(payload) < MIN_SOURCE_RECORDS:
        return False, [], f"entry count {len(payload)} < {MIN_SOURCE_RECORDS}"

    normalized: List[Dict[str, str]] = []
    seen = set()
    for index, entry in enumerate(payload):
        if not isinstance(entry, Mapping):
            return False, [], f"entry {index} is not an object"
        country, capital = country_capital_from_entry(entry)
        if not country or not capital:
            return False, [], f"entry {index} has empty country or capital"
        key = (country.casefold(), capital.casefold())
        if key not in seen:
            seen.add(key)
            normalized.append({"country": country, "capital": capital})
    if len(normalized) < MIN_SOURCE_RECORDS:
        return False, [], f"normalized count {len(normalized)} < {MIN_SOURCE_RECORDS}"
    return True, normalized, "passed"


def canonical_json_bytes(records: Sequence[Mapping[str, str]]) -> bytes:
    """Serialize normalized source facts deterministically."""
    sorted_records = sorted(
        records,
        key=lambda item: (item["country"].casefold(), item["capital"].casefold()),
    )
    return (
        json.dumps(sorted_records, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def safe_source_name(name: str) -> str:
    """Return a deterministic filesystem-safe source label."""
    return re.sub(r"[^a-z0-9_-]+", "_", name.lower()).strip("_")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run() -> int:
    """Acquire, validate, normalize, and pin the first valid independent source."""
    sources = (
        ("restcountries_all_fields", REST_ALL_URL),
        ("restcountries_independent_fields", REST_INDEPENDENT_URL),
        ("samayo_country_json_pinned", FALLBACK_URL),
    )
    failures: List[Dict[str, Any]] = []
    selected_result: HttpFetchResult | None = None
    selected_records: List[Dict[str, str]] = []

    for name, url in sources:
        print(f"[INFO] Fetching source: {name} {url}", flush=True)
        result = fetch_url(name, url)
        passed, normalized, reason = validate_payload(result)
        if not passed:
            failure = {
                "name": name,
                "url": url,
                "status": result.status,
                "bytes": len(result.body),
                "reason": reason if result.error is None else f"{reason}; {result.error}",
                "first_200_bytes": first_200(result.body),
            }
            failures.append(failure)
            print(
                f"[WARN] Source failed validation: {name} status={result.status} "
                f"bytes={len(result.body)} reason={failure['reason']}",
                flush=True,
            )
            continue
        selected_result = result
        selected_records = normalized
        break

    if selected_result is None:
        write_json(FAILED_FETCH_PATH, {"failures": failures})
        print(SEP, flush=True)
        print("[ERROR] All RB4 corpus sources failed validation.", flush=True)
        for failure in failures:
            print(
                f"[ERROR] {failure['name']} status={failure['status']} "
                f"bytes={failure['bytes']} reason={failure['reason']} "
                f"first200={failure['first_200_bytes']}",
                flush=True,
            )
        print(f"[INFO] Failure report: {FAILED_FETCH_PATH}", flush=True)
        return 2

    normalized_bytes = canonical_json_bytes(selected_records)
    normalized_sha256 = sha256_bytes(normalized_bytes)
    source_filename = (
        f"country_capitals_{safe_source_name(selected_result.name)}_"
        f"{normalized_sha256}.json"
    )
    source_path = SOURCE_DIR / source_filename
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(normalized_bytes)
    source_file_sha256 = sha256_file(source_path)
    if source_file_sha256 != normalized_sha256:
        raise RuntimeError("pinned source hash changed during write")

    selected_is_fallback = selected_result.url == FALLBACK_URL
    manifest = {
        "run_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selected_source": {
            "name": selected_result.name,
            "url": selected_result.url,
            "source_commit": FALLBACK_COMMIT if selected_is_fallback else None,
            "http_status": selected_result.status,
            "response_bytes": len(selected_result.body),
            "response_sha256": sha256_bytes(selected_result.body),
            "normalized_records": len(selected_records),
        },
        "source_file": str(source_path.relative_to(REPO_ROOT)),
        "source_file_sha256": source_file_sha256,
        "source_license": {
            "name": "MIT" if selected_is_fallback else "source-specific",
            "url": FALLBACK_LICENSE_URL if selected_is_fallback else "",
            "redistribution_permitted": selected_is_fallback,
        },
        "validation": {
            "http_status_200": selected_result.status == 200,
            "body_over_5_kib": len(selected_result.body) > MIN_RESPONSE_BYTES,
            "json_list_at_least_180": len(selected_records) >= MIN_SOURCE_RECORDS,
            "all_country_and_capital_non_empty": True,
            "top_level_error_absent": True,
        },
        "failed_sources": failures,
        "claim_status": (
            "PINNED FACT SOURCE ONLY. This file is not an RB4 role-binding corpus, "
            "contains no independently sourced construction families or ambiguity "
            "labels, and does not support a model-generalization claim."
        ),
    }
    write_json(MANIFEST_PATH, manifest)

    print(SEP, flush=True)
    print("[INFO] RB4 source acquisition completed.", flush=True)
    print(
        f"✓ Source passed validation: {selected_result.name} "
        f"records={len(selected_records)}",
        flush=True,
    )
    print(f"✓ Pinned source: {source_path}", flush=True)
    print(f"✓ Pinned source SHA-256: {source_file_sha256}", flush=True)
    print(f"[INFO] Manifest: {MANIFEST_PATH}", flush=True)
    print(SEP, flush=True)
    return 0


def main() -> int:
    """CLI entry point."""
    try:
        return run()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
