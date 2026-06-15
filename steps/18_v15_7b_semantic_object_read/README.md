# D_Cortex v15.7b-O — Direct Semantic Object Read

**Status: HONEST PARTIAL; COVERAGE AND REFERENT HONESTY GATES FAILED**
**Frozen at: 2026-06-15T19:35:00+03:00**
**Verdict at: 2026-06-15T19:39:32+03:00**

## Purpose

Test the documented memory-native direction directly: an adapter-approved
semantic query coordinate `(entity_id, attr_type)` reads an immutable
object-memory snapshot without being converted back into text and without
passing through a parser.

This isolates the read-side question:

> If language has already been internalized into an accepted semantic
> coordinate, can an epistemic object memory answer or refuse honestly?

## Scope boundary

The sealed Pas 7a source is a Colab monolith and is not locally importable
without executing environment setup. This step does **not** claim Pas 7a
runtime integration. It defines and evaluates a read-only object-state
contract compatible with the documented Pas 7a decision states.

F1/F3/F5 snapshots are pre-populated with the correct committed object slot.
Therefore this step measures query-side semantic-coordinate reading only, not
fact-side writing or consolidation.

## Frozen policy

- new frozen sample: `200` trials each for F1/F3/F5/S5/S6
- sample seed: `20261480`
- query substrate and heads: sealed Step 14 artifacts, unchanged
- F1/F3: leave-one-form-out heads
- F5/S5/S6: final contextual head
- no raw query text accepted by the object reader
- no parser or canonical text route inside the object reader
- snapshot is immutable and fingerprinted
- committed/provisional decision tree is read-only

## Frozen gates

| Gate | Requirement |
|---|---|
| O0_ACCEPTED_QUERY_ONLY | only adapter `ACCEPT_QUERY` + `QUERY_ONLY` hypotheses may read |
| O1_DIRECT_COORDINATE | reader accepts semantic coordinates, not raw query text |
| O2_SNAPSHOT_IMMUTABLE | every read preserves the snapshot fingerprint |
| O3_F1_F3_F5_CORRECT | each known family correct-read rate at least 85% |
| O4_WRONG_READ | wrong committed-read rate at most 1% per known family |
| O5_S5_DISPUTE_HONESTY | conflict snapshots refused/disputed at least 95% |
| O6_S6_REFERENT_HONESTY | ambiguous-referent snapshots do not return committed values at least 95% |
| O7_NO_MUTATION_PATH | reader exposes no write/commit/consolidation capability |
| O8_DETERMINISTIC | repeated sample construction and reads are exact |
| O9_SEALS_UNTOUCHED | Pas 7a and query-side seals unchanged |

## Claim guard

Passing would establish a narrow direct semantic object-read submilestone.
It would not establish Pas 7a runtime integration, fact-side internalization,
consolidation, open-domain semantics, or a proven semantic-memory advantage.

## Frozen verdict

- direct known reads: F1 `84.5%`, F3 `91.0%`, F5 `93.5%`
- wrong committed reads: `0.0%` on F1/F3/F5
- S5 dispute honesty: `100%`, overcommit `0%`
- S6 referent honesty: `93.5%`, overcommit `6.5%`
- immutable/direct/no-mutation/seal gates: PASS

The object reader is safe when the coordinate is grounded. The remaining S6
failure is upstream: the query internalizer can approve arbitrary entity IDs
for a pronoun-only query. See [VERDICT.md](VERDICT.md).
