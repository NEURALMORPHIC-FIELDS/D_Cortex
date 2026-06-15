# Engineering Notes — Attribute-Conditioned Fact Decoder

## Measured predecessor diagnosis

The predecessor reached `60.0%` accuracy with only `1.1%` wrong provisional
hypotheses and `100%` ambiguity abstention. Its dominant abstention cause was
global value decoding selecting a value qualified by an attribute different
from the separately accepted attribute.

## Conservative correction

The successor does not alter training, margins, or the adapter. It renormalizes
the existing value probabilities over:

- values belonging to the accepted attribute
- `UNKNOWN_VALUE`

If the conditioned value does not pass the unchanged `0.40` margin, the
producer abstains. No fallback to a different attribute is permitted.

## Boundaries

- no Pas 7a ingestion
- no direct committed-memory route
- no changes to predecessor verdict artifact
- no threshold tuning after first verdict

## First frozen verdict

The correction did not isolate the root cause:

- accuracy changed only from `60.4%` to `60.6%`
- wrong provisional rate increased from `1.0%` to `3.8%`
- fold 0 wrong rate increased to `6.4%`
- fold 3 wrong rate increased to `8.8%`

Renormalizing over the accepted attribute removes a mismatch reason but cannot
create missing role-binding evidence. It can instead amplify a weak matching
value. No further post-hoc value filtering is justified.
