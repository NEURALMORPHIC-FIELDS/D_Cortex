# VERDICT — D_Cortex v15.7b-RB0 Non-Trivial Role-Binding Benchmark

**Status: SEALED NON-TRIVIAL BENCHMARK**
**Verdict timestamp: 2026-06-15T19:49:55+03:00**
**Result: 8 PASS, 0 FAIL**

## Frozen result

| Baseline | Known exact | Known coverage | Ambiguous overcommit |
|---|---:|---:|---:|
| ordered first occurrence | `25.0%` | `100%` | `100%` |
| minimum distance | `37.1%` | `100%` | `100%` |
| lexical Cartesian | `0.0%` | `100%` | `100%` |
| safe abstain | `0.0%` | `0.0%` | `0.0%` |

All 2,000 record texts are unique. Known records contain exactly two entities,
two same-attribute values, and one one-to-one truth mapping. Ambiguous records
have no committed truth mapping.

## Evidence

- current-session verdict:
  `runs/semantic_role_binding_benchmark/results/verdict.json`
- verdict SHA-256:
  `c4dcd47d471d679fa20e78e178943599d2cda383e9fbcc23b5bcde39fe1bb876`
- frozen sample:
  `runs/semantic_role_binding_benchmark/results/sample.json`
- sample file SHA-256:
  `2c4a2dd117535b6ee7929bbb1c9882eddd95ab50fd97c4b1a343a9c196fd3625`
- semantic sample hash:
  `7e1d681c84ceb728fa92cf04e7c463605fd1e9a2af720e01875de45d843a956a`

## Claim guard

Supported:

- lexical inventory and position alone do not solve this controlled benchmark
- honest abstention alone cannot provide capability
- one learned role-binding experiment is justified

Not supported:

- semantic role-binding capability
- fact-side memory improvement
- open-domain proof
