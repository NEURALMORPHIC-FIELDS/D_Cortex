# -*- coding: utf-8 -*-
# Empirical hard-gate probe: can the SEALED symbolic organ store and return an
# arbitrary OPEN-STRING value faithfully? Organ ONLY. No Qwen, no GPU.

import sys
import traceback

print("=" * 70, flush=True)
print("PROBE: open-string round-trip through the sealed symbolic organ", flush=True)
print("=" * 70, flush=True)

try:
    from integration.organ_client import OrganClient, FOUND_COMMITTED
except Exception as exc:  # noqa: BLE001
    print(f"[HARD-BLOCKER] could not import OrganClient: {exc!r}", flush=True)
    traceback.print_exc()
    sys.exit(2)

try:
    oc = OrganClient()
except Exception as exc:  # noqa: BLE001
    print(f"[HARD-BLOCKER] OrganClient() failed to construct (organ load): {exc!r}", flush=True)
    traceback.print_exc()
    sys.exit(3)

print("[INFO] OrganClient constructed OK.", flush=True)
print(f"[INFO] attribute vocabulary keys: {sorted(oc.attr_values.keys())}", flush=True)
for k in sorted(oc.attr_values.keys()):
    print(f"[INFO]   {k}: {oc.attr_values[k]}", flush=True)
print(f"[INFO] known_entities (first 5): {oc.known_entities[:5]}  (total={len(oc.known_entities)})", flush=True)
print("-" * 70, flush=True)

# Open-string facts we WANT to store faithfully (patent-style).
ENTITY = oc.known_entities[0] if oc.known_entities else "widget"
OPEN_FACTS = [
    ("patent_number", "EP25216372.0"),
    ("filing_date",   "2025-12-19"),
    ("applicant",     "Fragmergent Technology S.R.L."),
]

# Also a control: a value that IS in the closed vocabulary, to prove the organ
# itself works for in-vocab values (so a False verdict is about openness, not breakage).
in_vocab_attr = "color"
in_vocab_val = oc.attr_values.get("color", ["red"])[0]
CONTROL = [(in_vocab_attr, in_vocab_val)]

print("[STEP] begin_episode()", flush=True)
ep = oc.begin_episode()
print(f"[INFO] episode = {ep}", flush=True)

print("-" * 70, flush=True)
print("[STEP] write_fact for OPEN-STRING values:", flush=True)
for attr, val in OPEN_FACTS:
    res = oc.write_fact(ENTITY, attr, val)
    print(f"  write_fact(entity={ENTITY!r}, attr={attr!r}, value={val!r})", flush=True)
    print(f"    -> is_attribute={oc.is_attribute(attr)} is_value={oc.is_value(attr, val)}", flush=True)
    print(f"    -> result={res}", flush=True)

print("-" * 70, flush=True)
print("[STEP] write_fact for CONTROL in-vocab value:", flush=True)
for attr, val in CONTROL:
    res = oc.write_fact(ENTITY, attr, val)
    print(f"  write_fact(entity={ENTITY!r}, attr={attr!r}, value={val!r})", flush=True)
    print(f"    -> is_attribute={oc.is_attribute(attr)} is_value={oc.is_value(attr, val)}", flush=True)
    print(f"    -> result={res}", flush=True)

print("-" * 70, flush=True)
print("[STEP] end_episode() (Pas7a consolidation)", flush=True)
try:
    rep = oc.end_episode()
    print(f"[INFO] end_episode returned: {type(rep).__name__}", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"[WARN] end_episode raised: {exc!r}", flush=True)
    traceback.print_exc()

print("-" * 70, flush=True)
print("[STEP] read back each value via query():", flush=True)
for attr, val in OPEN_FACTS + CONTROL:
    reply = oc.query(ENTITY, attr)
    exact = (reply.value == val)
    print(f"  query(entity={ENTITY!r}, attr={attr!r})  wanted={val!r}", flush=True)
    print(f"    -> status={reply.status!r} value={reply.value!r} exact_match={exact}", flush=True)
    print(f"    -> trace={reply.trace}", flush=True)

print("-" * 70, flush=True)
# Direct probe of the raw bank read to expose the integer index representation.
print("[STEP] raw bank.read_attribute() to expose internal representation:", flush=True)
try:
    bank = oc._bank
    for attr, val in OPEN_FACTS + CONTROL:
        raw = bank.read_attribute(ENTITY, attr)
        print(f"  bank.read_attribute(entity={ENTITY!r}, attr={attr!r}) -> {raw!r}  (type of [1]={type(raw[1]).__name__})", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"[WARN] raw bank read failed: {exc!r}", flush=True)
    traceback.print_exc()

print("=" * 70, flush=True)
print("PROBE DONE", flush=True)
print("=" * 70, flush=True)
