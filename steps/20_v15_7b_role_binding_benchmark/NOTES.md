# Engineering Notes — Non-Trivial Role-Binding Benchmark

## Why this comes before another model

The original neural-memory benchmark failed non-triviality because a cloze
prefix let lexical retrieval solve the task. The first fact-side classifier
also operated on records containing one entity and one value, so a future
span extractor could appear successful without learning binding.

This benchmark forces a one-to-one semantic assignment between two entities
and two values while keeping the lexical inventory identical across competing
mappings.

## Baseline intent

- ordered baseline tests first-occurrence pairing
- minimum-distance baseline tests positional proximity
- Cartesian baseline tests token inventory without binding
- safe abstain tests whether honesty alone can masquerade as capability

Only after these baselines fail the frozen non-triviality gates is investment
in a learned role binder justified.

## Frozen verdict interpretation

The best non-abstaining baseline was minimum-distance pairing at `37.1%`
exact-known. Every non-abstaining lexical baseline overcommitted on `100%` of
ambiguous records. Safe abstention avoided overcommit but provided `0%` known
coverage.

The benchmark therefore removes the measured token-overlap shortcut and
justifies one learned role-binding experiment.
