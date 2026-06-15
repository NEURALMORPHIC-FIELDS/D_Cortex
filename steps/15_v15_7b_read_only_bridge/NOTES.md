# Engineering Notes — Read-Only Semantic Query Bridge

## Architectural decision

The bridge is intentionally a pure routing component. It translates an
already-approved semantic query interpretation into a canonical query string.
It cannot inspect or mutate memory and cannot decide whether a memory answer is
safe.

This preserves the documented boundary:

- semantic producer: proposes an interpretation
- conservative adapter: approves or rejects it
- read-only bridge: selects a canonical read route
- memory reader: owns read semantics and honesty
- Pas 7a: owns committed/provisional mutation and remains sealed

## Expected diagnostic value

F1, F3, and F5 isolate whether semantic query routing improves actual memory
recall. S5 and S6 isolate whether the target read path can refuse ambiguous
answers.

The current trained neural-memory answer head is known to emit one token. It
has no committed/provisional/disputed status. Therefore S5/S6 may block the
integration verdict even if routed recall improves. Such a failure is a
reader-honesty limitation, not a bridge failure, and must be reported as such.

## Frozen source hashes

- Pas 7a:
  `25b4906ecc05a6b51b10902e54332a0ec2b26c4c622aa4e6ee74bd4961369aa3`
- semantic adapter:
  `719afbe8359af4ee98006e0fa19f169a0a40f22ee0cc41e5dc1459d6e07e605e`
- semantic producer:
  `24797a121c90f78161518362f0401bbc1b161546e87bac194adb4144f4aa67d0`
- contextual evaluator:
  `bab8ec1268d5000d5a9da5ae5946d29946822efa9a6b0b3ce53be59052631b57`

## Frozen implementation scope

New runtime component:

- `dcortex/semantic_query_bridge.py`

New verification:

- `tests/test_semantic_query_bridge.py`
- `scripts/semantic_bridge_end_to_end.py`

Allowed existing-file changes:

- public export in `dcortex/__init__.py`
- integration registry update
- append-only project documentation and memory updates

Forbidden:

- modification of Pas 7a source
- modification of sealed semantic adapter/producer/contextual evaluator
- direct memory write or commit path
- threshold changes after the first end-to-end verdict

## Post-verdict diagnosis

The first frozen verdict showed that accurate semantic query routes do not
improve the trained neural-memory answer path. F3/F5 contain only one written
fact, so their unchanged `36.0-36.5%` recall isolates a downstream
value-emission/generalization limitation. S5/S6 also expose the absence of a
read-time refusal status.

A development-only cloze-address diagnostic did not justify another routing
iteration. The next experiment therefore moves to conservative fact-side
semantic hypotheses, which may enter only the adapter's provisional channel.
