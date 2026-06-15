# Engineering Notes — Leave-One-Syntax-Family-Out Role Binding

## Why this test precedes integration

RB2 reached 100% on held-out texts, but every syntax family was represented in
training. Integrating that result into a memory writer would risk treating
template recognition as semantic role-binding capability.

RB3 changes no architecture. It changes only data separation:

- one complete known syntax family is absent from training and validation
- every known record is tested exactly once across four held-out folds
- ambiguity calibration remains separate and validation-only

The aggregate lexical baseline is computed over the same 1,600 held-out known
records, preserving RB0 non-triviality.

## Measured diagnosis

| Held-out family | Exact | Wrong | Abstained |
|---|---:|---:|---:|
| RB1 | `62.8%` | `24.3%` | `13.0%` |
| RB2 | `39.2%` | `60.8%` | `0.0%` |
| RB3 | `79.3%` | `20.8%` | `0.0%` |
| RB4 | `46.5%` | `20.0%` | `33.5%` |

Every fold fits its seen-family validation set perfectly and preserves
ambiguity honesty, yet unseen construction behavior varies sharply. The
remaining bottleneck is construction diversity and relation generalization,
not optimizer convergence or role-mask coverage.

No further architecture iteration or memory integration is justified on RB0
alone. The next gate must be data-only and independently sourced.
