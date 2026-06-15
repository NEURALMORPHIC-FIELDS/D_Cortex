# Engineering Notes — Token-Level Role-Conditioned Binder

## Why this architecture

RB1 learned RB5 ambiguity perfectly but remained near chance on identity versus
swapped mappings in every known family. The observed failure is relational,
not lexical coverage or optimization.

RB2 therefore removes only the lossy pooling bottleneck:

- source token states remain frozen D_Cortex outputs
- exact candidate inventory marks mentions but does not supply the truth mapping
- the same learned sequence scorer evaluates identity, swapped, and unresolved
- no handwritten interpretation of `not`, `latter`, `former`, or other relation
  words is permitted

## Safety

The sequence scorer can only propose a complete mapping. The existing
conservative binder still decides whether the calibrated margin permits
emission, and the semantic adapter remains the only provisional destination.

## Measured result and remaining limit

The role-conditioned scorer reached exact binding on every held-out text in the
unchanged RB1 test split while preserving perfect ambiguity abstention.

This closes the measured pooling bottleneck, but all four known syntax families
were present during training. The result does not distinguish relational
generalization from family-template learning. A leave-one-syntax-family-out
cross-validation is therefore required before any memory-ingestion experiment.
