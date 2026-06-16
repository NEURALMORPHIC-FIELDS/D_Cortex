# Seed-variance localization (vnext3 structural certification)

Aggregate held-out exact across 20 seeds: median 99.3%, min 97.0%, max 99.9%, std 0.008.

- Lowest seed: #6 exact 97.0%; highest seed: #13 exact 99.9%.
- Construction with the highest cross-seed std: **concessive** (std 0.024).
- Per-construction cross-seed std: appositive=0.000, concessive=0.024, copular=0.000, embedded=0.015, long_gap=0.001, relative=0.011.
- No-memory (zeroed-content) control exact: median 51.6% (binder uplift +47.7%).

Conclusion: cross-seed variance is already low (std <= 2pp). The residual spread is localized to the **concessive** construction; no pipeline instability (init / abstain-calibration / wrong-mapping spike) dominates.
