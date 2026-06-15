# Engineering Notes — Independent Role-Binding Corpus Gate

## Why development stops at data

RB2 achieved perfect held-out-text accuracy when syntax families were seen.
RB3 then measured only `56.9%` exact and `31.4%` wrong on unseen syntax
families. This is direct evidence that construction diversity, not another
threshold or post-hoc filter, is the next bottleneck.

Generating more variants from the same four templates would not create an
independent test. A new corpus must preserve source provenance and explicit
construction separation before model work resumes.

## Safety consequence

RB2 remains a sealed controlled component, but it is not authorized for
automatic Pas 7a ingestion. Its outputs remain provisional-only in controlled
seen-syntax scope.
