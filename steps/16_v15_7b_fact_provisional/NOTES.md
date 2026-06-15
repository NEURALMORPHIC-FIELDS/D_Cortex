# Engineering Notes — Fact-Side Provisional Producer

## Why this successor is justified

The query-side internalizer is accurate, but the frozen bridge-readiness
verdict showed no end-to-end recall uplift. F3/F5 each had one written fact,
yet recall remained only `36.0-36.5%`. This isolates a downstream fact/value
path limitation that query routing cannot repair.

The documented v15.7b direction permits semantic facts only as provisional
hypotheses. Therefore the next experiment tests fact interpretation while
preserving the Pas 7a commitment boundary.

## Value identity

The value classifier uses attribute-qualified IDs such as `color:red` and
`state:awake`. An emission is allowed only when the selected value's attribute
matches the independently selected attribute. The adapter receives the plain
value ID only after this consistency check passes.

## Frozen ambiguity families

Evaluation ambiguity forms are absent from training:

1. two conflicting values for one entity and attribute
2. two entities sharing one value statement
3. non-fact discourse sentences

Any emitted hypothesis on these families is an honesty failure.

## Frozen boundaries

- new module: `dcortex/semantic_fact_producer.py`
- new evaluator: `scripts/semantic_fact_curriculum.py`
- new tests: `tests/test_semantic_fact_producer.py`
- no modification of Pas 7a
- no modification of sealed adapter, query producer, contextual evaluator, or
  read-only bridge
- no committed-memory write
