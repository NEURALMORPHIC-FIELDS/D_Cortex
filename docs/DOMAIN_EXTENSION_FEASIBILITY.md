# D_Cortex domain extension: real-facts feasibility map (patents)

Engineering register. MEASURED, deterministic (Qwen-4bit greedy), one domain, 15 real
source-pinned patents, single machine, NOT production. Source of every number below:
`runs/domain_extension/results/verdict.json` (script `scripts/certify_domain_extension.py`).
`dcortex/` and `steps/13` are byte-identical (loaded read-only); open values reach the
organ only through the reference-token adapter `integration/domain_adapter.py`.

## Verdict: D_CORTEX_DOMAIN_PARTIAL_CAPACITY_BOUND

This is a feasibility/risk campaign; the deliverable is a map of where the real domain
breaks, not a PASS. It is PARTIAL, as expected. The integrity gates pass, but ONLY on the
small subset of facts the organ can physically hold. Leading with what breaks.

## What BREAKS (lead)

1. **Capacity is the dominant blocker (value_storage = 34 / 89 facts unstorable).** The
   sealed organ is a closed-vocabulary index store: 4 attributes (color/size/location/state)
   and 37 total value tokens (15/4/10/8). A patent has 6 schema attributes, so:
   - `filing_date` and `title_keyword` have NO organ slot at all (only 4 slots exist) ->
     30 facts (2 attributes x 15 patents) cannot be stored, period.
   - `applicant` maps to `location` (10 tokens) and OVERFLOWS at the 11th distinct value ->
     4 more facts abstain (measured: 14 distinct applicants, 10 stored, 4 overflow).
   - `patent_number` maps to `color` (15 tokens) and saturates EXACTLY at 15 patents; a 16th
     would overflow.
   The closed 4-attribute / 37-token vocabulary cannot represent an open domain. The adapter
   correctly ABSTAINS on overflow (never collides two open strings onto one token), so this
   is a hard ceiling, not a corruption.

2. **Update friction: a single fact update is not reflected.** Writing a real lifecycle
   `legal_status: pending -> granted` does NOT flip the stored value in one episode. The
   sealed consolidator is conservative: the conflicting write goes to `FOUND_DISPUTED`, then
   transiently `NONE_ATTRIBUTE`, and only after 3 reinforcement episodes does `granted`
   promote to `FOUND_COMMITTED`. Trajectory (real, measured):
   `ep1 pending=COMMITTED -> ep2 granted=DISPUTED -> ep3 granted=NONE_ATTRIBUTE -> ep4
   granted=COMMITTED -> ep5 granted=COMMITTED`. This is integrity-preserving (it never
   silently overwrites and flags the dispute) but means a real status change needs repeated
   assertion, not a single update.

3. **Extraction was not stress-tested.** `source_text` is a CLEAN bibliographic sentence
   assembled from the verified fields with the answer stated verbatim (e.g. "filed on
   1998-01-09", "IPC section G"). It is NOT a raw patent abstract / claims / OCR. The
   extraction numbers are therefore an UPPER BOUND and say nothing about the genuinely hard
   real-document path. Moreover the 6 apparent open-attribute "misses" are gold-normalization
   strictness, not model errors: Qwen returned valid full surface forms ("National Security
   Agency" vs canonical "NSA", "Sealed sandwich" vs "crustless sandwich"). Real extraction on
   this clean text is ~100%.

## What WORKS (on the storable subset)

- **Organ integrity holds on real records: `wrong_commit = 0` over 55 committed reads.** Every
  gold fact the organ accepted reads back as the exact gold open value (decoded through the
  reference-token adapter).
- **Veto blocks real, confident, WRONG model assertions.** On 6 uncovered queries (no fact in
  memory) the veto-controlled path abstained 6/6 with 0 grounded leak. Asked the same with the
  patent's real identity, the RAW unconstrained model confidently asserted a concrete value on
  5/6, and at least 3 are verifiably WRONG against the source: applicant "W. L. Gore & Associates"
  for US4733665A (real assignee: Expandable Grafts Partnership), filing date "October 16, 1957"
  for the LEGO brick US3005282A (real: 1958-07-28), filing date "March ..., 1997" for US6004596A
  (real: 1997-12-08). The veto converted all of these to abstain (5 -> 0). HONEST scoping: the
  detector still counts 5/6 with one truncated/ambiguous case (pat15), so the genuine-hallucination
  count is ~4-5/6; and the 55/55 "covered decode-faithful" figure is faithful BY CONSTRUCTION of
  constrained decoding (it proves the decode plumbing, not model restraint). `entity_resolution=0`
  is ASSERTED (entity is the deterministic corpus key), not measured.
- **The reference-token adapter works:** an open string ("US6285999B1") round-trips through
  the REAL arbiter (RoMR at write, Pas7a at end_episode) by standing in as a sealed-vocab
  reference word (color index 0 = "red"), with the open<->token table held in `integration/`.
  No edit to `dcortex/` or `steps/13`.

## Part 0 (hard gate): G_VALUE_OPEN

Confirmed from two independent sources (a direct code read and an empirical probe): the organ
stores a value as an integer index into a frozen closed vocabulary (`value_idx` is "PRIMARY
TRUTH", `code.py:4378`); `write_fact` rejects any out-of-vocab value (`out_of_vocabulary`),
`read_attribute` returns an int. Native open-string round-trip = FALSE. Via the adapter =
TRUE. There is no dynamic value-registration in the write chain; an open domain is only
representable through the reference-token indirection, within the capacity ceiling above.

## Gates (measured)

| Gate | Result | Note |
|------|--------|------|
| G_VALUE_OPEN | PASS | native fails, adapter round-trips |
| G_EXTRACT_REAL closed | 1.00 (bar 0.85) | legal_status 1.0, ipc_section 1.0 (on clean assembled text) |
| G_EXTRACT_REAL open | 0.90 (bar 0.60) | patent_number 1.0, filing_date 1.0, applicant 0.8, title 0.8; misses are normalization strictness; UPPER BOUND |
| G_ORGAN_REAL | wrong_commit = 0 | 55 reads; 4 capacity overflows abstained; update needs 3 reinforcements; RoMR inherited (not re-exercised) |
| G_HALLUC | controlled grounded-on-uncovered = 0 | raw model asserted 5/6 (>=3 verifiably wrong); veto reduced to 0. Covered 55/55 decode-faithful is by-construction, not model restraint |
| G_NOBYPASS | PASS | 0 bypass, 6/6 uncovered abstain (real reads + structural no-slot) |
| G_ERROR_MAP | value_storage 34, extraction_open 6, organ_commit 0, extraction_closed 0, entity_resolution 0 | DOMINANT: value_storage |

## Corpus provenance

15 real patents fetched live from Google Patents (per-record `source_url` in
`integration/patent_corpus.py`), IPC sections A,B,C,D,F,G,H exercised (E not obtained with a
confirmed primary class, so not claimed). One field (IPC of US7479949B2) did not render and is
left empty / excluded from gold (abstention case). The owner patent **EP25216372.0 is absent**:
it returned HTTP 404 and has no public bibliographic record (normal for an EP application filed
late 2025, unpublished until ~18 months after priority). No value was fabricated.

## The honest takeaway (and the next bet)

Entity, value, and veto machinery are sound, and an open string CAN be carried through the
sealed organ via a reference-token adapter without touching the organ. But the organ as sealed
is a closed 4-attribute / 37-token store: it cannot HOLD an open real domain. The dominant
blocker is not extraction and not entity resolution; it is the organ's value/attribute
**capacity**. Making D_Cortex hold real domain facts requires extending the symbolic
vocabulary of the organ itself (more attributes, open value representation) - i.e. editing or
retraining the sealed substrate - which is the large, unproven bet this campaign was meant to
scope, and which is explicitly out of scope here (organ byte-identical). The adapter is a
bridge for a handful of values, not a path to an open domain.
