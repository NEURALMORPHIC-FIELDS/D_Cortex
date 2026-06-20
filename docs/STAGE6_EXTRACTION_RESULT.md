# Stage 6 FREE-TEXT EXTRACTION - MEASURED: STAGE_6_SUBSTRATE_LIMITED (varied-phrasing binding is a substrate property)

Cert: scripts/certify_stage6_extraction.py | verdict: runs/stage6_extraction/results/verdict.json
External extractor on the FROZEN ckpt_multiobject substrate; VARIED phrasing over a closed vocab; DOUBLE
held-out (unseen entities AND unseen phrasings); 5 seeds. VERDICT STAGE_6_SUBSTRATE_LIMITED.

## LEAD WITH THE DANGEROUS METRIC (honest extraction = wrong-binding ~0, not just high recovery)
| gate                                          | result            |
|-----------------------------------------------|-------------------|
| G_WRONG_VALUE_BINDING (entity cross-bound; bar <=0.02) | 0.38 FAIL (38% fabrication) |
| G_VALUE_BINDING_PHRASE (varied phrasing; bar >=0.85)   | 0.557 FAIL        |
| G_RELATION_DIRECTION_WRONG (wrong target/direction; bar <=0.05) | 0.52 FAIL |
| G_RELATION_BINDING (subject->target; bar >=0.75)       | 0.48 FAIL         |
| G_RELATION_ABSTAIN_BROKEN (broken -> abstain; bar 0.80)| 0.96 (works)      |
Even a trained external extractor recovers value binding from varied phrasing at only ~0.56 (held-out
entities AND phrasings), with ~38% CROSS-BINDING (the entity bound to the SIBLING's value) - that is
fabrication, not honest extraction. Relations are worse (0.48 binding, 0.52 wrong-direction).

## What this proves (the decisive answer)
The frozen substrate (trained ONLY on "The X is Y." TEMPLATES) does NOT expose value/relation bindings
from VARIED phrasing. Step 2 made value-separability generalize across ENTITIES, but with the phrasing
fixed; it never trained across PHRASINGS, so varied surface structure is OUT-OF-DISTRIBUTION for the
substrate. An external readout cannot recover what the substrate does not represent. Extraction-binding
from varied phrasing is a SUBSTRATE property, not an external-module property. This is Stage-I-redux at
the phrasing level (Stage I failed multi-fact binding on the frozen base; here the frozen base fails
varied-phrasing binding).

## Honest nuance (the honesty mechanism DOES transfer)
G_RELATION_ABSTAIN_BROKEN = 0.96: when the relation points to a non-stored target, the predicted key
does not content-address any stored slot (low confidence) -> abstain. So the HONESTY layer (5e's
confT-based abstain) transfers even here - the front-end refuses broken relations. Only the BINDING is
substrate-limited; the refusal-when-uncertain is not. The extractor fails by binding WRONG (fabrication
0.38), but the downstream honesty would still catch a followed-but-broken pointer.

## The next move (clear, scoped - not a mystery)
SUBSTRATE FINE-TUNE ON VARIED PHRASING - the Step-2 move, one level up. Step 2 fine-tuned the encoder so
value-binding generalizes across ENTITIES (templated). The analogous campaign: fine-tune the substrate
on VARIED PHRASING so the binding survives surface variation, then re-run this extractor cert. The
decisive question becomes whether a phrasing-trained substrate exposes the binding (as the
entity-trained substrate exposed it across entities at 0.92). If yes, extraction generalizes and the
autonomy loop closes; if not, varied-phrasing binding needs a deeper representational change.

## Where this leaves the program
- The MECHANISM arc (Stage 5 -> 5e) is COMPLETE and proven: operate over persisted memory (single-step
  + graph traversal), honestly (wrong_commit=0 extended to multi-hop), on one substrate, validity-gated.
- The EXTRACTION front-end from FREE TEXT is NOT yet closed: the templated-trained substrate does not
  expose varied-phrasing binding. The precise, scoped next step is a substrate fine-tune on varied
  phrasing. This negative localizes the work exactly.
Single architecture, closed vocab, small synthetic, double held-out - mechanism diagnosis, not generality.
