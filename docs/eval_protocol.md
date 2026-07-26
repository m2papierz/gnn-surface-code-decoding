# Evaluation Protocol — Pre-registered Stopping Rule

This document pre-registers the statistical decision rules for all evaluation
runs in this project. It must be committed before the first full evaluation
run (T13). No parameter in this document may be changed after evaluation
begins without recording the change as a protocol amendment with justification.

---

## 1. Primary Decision Tool

**McNemar's test** on the per-shot disagreement matrix between the GNN decoder
and the baseline (MWPM). All decoders run on identical frozen shots, making
the paired test the correct and most powerful choice.

- Significance level: **α = 0.05** (two-sided) per (d, p) point.
- No multiple-testing correction across (d, p) points. Each point is an
  independent physical setting; the result is a table of standalone claims,
  not an omnibus hypothesis.

Wilson 95% confidence intervals are reported on every LER value for display
(error bars on plots) but **never** used to make parity decisions. Overlapping
intervals do not imply parity; non-overlapping intervals do not decide
superiority — McNemar does both.

---

## 2. Adaptive Early Stopping

Evaluation proceeds in batches. Early stopping prevents wasting shots when the
outcome is already clear.

### Boundary

**Haybittle-Peto**: a fixed interim boundary of **α_stop = 0.001** at every
interim check, with the final analysis conducted at **α = 0.05**.

Justification: the Haybittle-Peto boundary spends negligible α at each interim
look (≤ 0.0005 per look at the 0.001 level). The final-analysis α remains
essentially 0.05 regardless of the number of interim checks. No α-spending
function is required.

### Timing

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Check interval | 10,000 shots | Balances computation cost vs granularity |
| Minimum shots before first check | 10,000 | Prevents decisions on trivially small samples |
| Maximum shots per (d, p) point | 1,000,000 | Bounds total eval time; exceeds error requirements at all settings |

### Procedure

1. Process shots from the frozen eval set in batches of 10,000.
2. After each batch (once cumulative shots ≥ 10,000):
   - Count logical errors under each decoder.
   - If **both** decoders have accumulated ≥ 100 logical errors:
     - Compute McNemar's test statistic on the cumulative disagreement matrix.
     - If p < 0.001 (interim boundary): **STOP** → outcome is
       `resolved-different`.
   - If either decoder has < 100 errors: **CONTINUE** (insufficient power).
3. At exhaustion of the frozen set or at 1,000,000 shots (whichever comes
   first):
   - If **both** decoders have ≥ 100 errors:
     - If McNemar p < 0.05: outcome is `resolved-different`.
     - If McNemar p ≥ 0.05: outcome is `resolved-parity`.
   - If either decoder has < 100 errors: outcome is `unresolved` (eval set
     undersized for this point — flag for regeneration with more shots).

---

## 3. Decision Outcomes

Every (d, p) evaluation point produces exactly one of three outcomes:

| Outcome | Meaning | Reported as |
|---------|---------|-------------|
| `resolved-different` | McNemar rejected H₀ at the specified level | Direction (which decoder wins) + LER values + Wilson CIs + McNemar statistic + p-value |
| `resolved-parity` | McNemar failed to reject at α=0.05 after sufficient data | LER values + Wilson CIs + McNemar statistic + p-value + shots consumed |
| `unresolved` | Insufficient logical errors (< 100) under one or both decoders | Flag: eval set must be regenerated with more shots before a claim is made |

"Parity" means the test lacked power to detect a difference at the given sample
size — not that the decoders are provably identical.

---

## 4. Minimum Error Requirements

| Requirement | Value | Rationale |
|-------------|-------|-----------|
| Minimum errors per decoder for a valid McNemar test | 100 | Ensures ≥ ~50 per off-diagonal cell; chi-squared approximation reliable |
| Frozen eval set target errors (under MWPM) | ≥ 400 | Comfortable power for detecting 10% relative LER differences |
| Frozen eval set floor errors (under MWPM) | ≥ 100 | Absolute minimum for a valid comparison |

Frozen eval sets (generated in T10) are sized per (d, p) point to meet the
400-error target under MWPM. The set generation script will report the actual
error count and flag any point below the 100-error floor.

---

## 5. Reporting Requirements

Every evaluation result reports:

- **Per-shot logical error rate (LER)**: errors / total_shots.
- **Per-round logical error rate (ε)**: 1 - (1 - LER)^(1/r), for cross-r
  comparability.
- **Wilson 95% confidence interval** on both LER and ε.
- **Shot count** (n_shots) and **error count** (n_errors) per decoder.
- **McNemar test statistic** and **p-value**.
- **Decision outcome**: one of the three outcomes above.
- **Shots consumed** before stopping (if early-stopped).

No rate is reported without an interval. No interval is reported without
visible counts.

---

## 6. Baselines

All baselines decode the same frozen shots as the GNN:

1. **PyMatching** (DEM-weighted MWPM) — primary comparison target.
2. **BP+OSD** (via CUDA-Q) — secondary baseline.

Parity decisions are made against PyMatching only. BP+OSD numbers are reported
for context but do not enter the McNemar comparison.

---

## 7. Evaluation Grid

| Distance | Rounds | Physical error rates (p) |
|----------|--------|--------------------------|
| 3 | 3 | 0.003, 0.005, 0.008, 0.01 |
| 5 | 5 | 0.003, 0.005, 0.008, 0.01 |
| 7 | 7 | 0.003, 0.005, 0.008, 0.01 |

Total: 12 (d, p) points evaluated independently.

---

## 8. Prohibitions

- **No threshold tuning on eval shots.** Decision thresholds are calibrated on
  the frozen validation set only; the eval set is never used to select or
  adjust any decoder parameter.
- **No peeking adjustments.** This document's parameters are fixed before the
  first eval run. Any change after eval begins requires a protocol amendment
  committed with explicit justification.
- **No post-hoc outcome reclassification.** A point resolved as "different"
  stays "different" even if a later model revision closes the gap.
- **No claim from interval overlap.** Overlapping Wilson CIs never justify a
  parity claim; only McNemar p ≥ α does.

---

## 9. Protocol Amendments

If any parameter in this document must be changed after the first evaluation
run:

1. The amendment is committed as a separate, clearly marked change to this
   file.
2. The commit message states the reason.
3. Any results produced under the old protocol are re-evaluated under the new
   protocol, or clearly marked as produced under different rules.
4. The Trurlic decision graph is updated (`update_decision` with
   `mode="supersede"`).
