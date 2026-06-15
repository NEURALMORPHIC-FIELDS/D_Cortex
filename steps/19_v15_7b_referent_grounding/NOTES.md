# Engineering Notes — Explicit Referent Grounding Guard

## Measured predecessor failure

Step 18 S6 overcommit was `13/200 = 6.5%`. Every overcommit came from an
arbitrary entity coordinate selected for a pronoun-only query:

- `crown`: 10
- `scroll`: 3

Neither entity appeared in the source query.

## Conservative correction

The guard accepts a semantic coordinate only when the normalized token
sequence of `entity_id` occurs contiguously in the existing hypothesis
`source_text`.

The source text is already audit evidence carried by the semantic hypothesis.
No new raw-text parameter enters object memory. No aliases are expanded.

## Boundaries

- does not improve semantic classifier coverage
- does not resolve pronouns
- does not modify the sealed adapter or query producer
- does not mutate object memory

## Frozen verdict interpretation

The guard removed every S6 committed answer on the new sample and preserved
all known-family and S5 rates. The relative uplift gate nevertheless failed
because the predecessor happened to reach `96.5%` S6 honesty on this sample,
leaving only `+3.5pp` available.

The threshold is not changed and the sample is not rerun. This branch is
stopped as an absolute safety improvement with an honestly failed frozen
milestone gate.
