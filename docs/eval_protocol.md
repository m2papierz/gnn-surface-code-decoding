# Evaluation Protocol — Pre-registered Stopping Rule

This document pre-registers the statistical decision rules for all evaluation
runs in this project. It must be committed before the first full evaluation
run. No parameter in this document may be changed after evaluation
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

Frozen eval sets are sized per (d, p) point to meet the
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

All baselines decode the same frozen shots as the GNN. A baseline takes only
its circuit: its configuration is fixed in `src/decoders.py` and its package is
pinned exactly in `pyproject.toml`, so a lock refresh cannot move a baseline.

| Baseline (report name) | Package, exact pin | Configuration |
|------------------------|--------------------|---------------|
| MWPM (`mwpm`) — primary comparison target | pymatching 2.3.1 | DEM with hyperedges decomposed into edges; one matching pass |
| Correlated matching (`correlated`) | pymatching 2.3.1 | Decomposed DEM; two-pass correlated matching |
| Belief-Matching (`belief_matching`) | beliefmatching 0.1.1, its BP from ldpc 2.4.1 | Decomposed DEM; product-sum BP, at most 20 iterations; BP's correction where BP converges, otherwise matching weighted by BP's posteriors |
| Tesseract (`tesseract`) | tesseract-decoder 0.1.1.dev20260802231159 | Undecomposed DEM; the package's default long-beam preset: beam 20 with beam climbing, 21 detector orders (seed 2384753), priority queue of at most 1,000,000 states |

MWPM and correlated matching decode every harness point. The eval config
chooses Belief-Matching (`include_belief_matching`) and Tesseract
(`include_tesseract`); a baseline the config asks for either decodes or the run
fails — none is skipped for a missing package.

Parity decisions are made against MWPM only. The other baselines' numbers are
reported for context but do not enter the McNemar comparison.

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

### 9.1 Amendment of 2026-10-06 — scaling diagnostics

Two rules, added for `scripts/diagnose_scaling.py`. Both were fixed as recorded
decisions and in code before the first diagnostics run, and no parameter of
either was changed after a result existed. Nothing in §1–§8 changes for any
comparison that involves the GNN, Belief-Matching or Tesseract.

**A. Baseline-only comparisons run the final analysis once, on every shot.**
When every decoder in a comparison is a classical baseline whose full-set
decode is cheap — MWPM against correlated matching — the frozen set is decoded
in full and only step 3 of §2 is applied: McNemar at α = 0.05, behind the
100-error floor of §4, with the three outcomes of §3. There are no interim
looks.

Reason: the interim boundary exists to stop paying for an expensive decoder.
Here it would save nothing, and the error rates would come from a truncated
sample whose size the comparison's own p-value chose. Those rates are not
display-only — they feed rule B and are the measured baseline rates that later
tasks read. One look spends less α than the sequence it replaces, so the
registered level holds. The comparison is oriented with correlated matching as
the reference and MWPM as the compared decoder, and the artifact names both
roles.

Decision: `baseline-only-diagnostics-decode-every-frozen-shot-and-run`.

**B. Sub-threshold is decided by a test between distances, per decoder.**
For one operation, one decoder and one physical error rate, the distances are
independent samples — different circuits, different shots — so the paired test
of §1 does not apply. The rule:

| Element | Value |
|---------|-------|
| Metric | Per-shot LER, a shot being wrong when any observable is wrong |
| Step | Each pair of consecutive distances d < d′ |
| Test per step | Pooled two-proportion z-test, two-sided, α = 0.05 |
| Validity floor | Smallest expected cell under the pooled rate ≥ 5 (Cochran) |
| Step verdict | `decreasing` if the test rejects and LER(d′) < LER(d); `increasing` if it rejects the other way; `unresolved` otherwise |
| `sub-threshold` | At least one step, and every step `decreasing` |
| `above-threshold` | At least one step, and every step `increasing` |
| `undetermined` | Anything else: mixed steps, an unresolved step, a single distance |

The rule is applied to each decoder separately, because a threshold belongs to
a decoder; which decoder's verdict governs a gate is stated where that gate is
defined. No correction is applied across steps or points: requiring every step
to reject is already conservative. The 100-error floor of §4 is not reused
here — it protects McNemar's off-diagonal cells, and a larger distance making
few errors is the evidence of being sub-threshold, not a lack of it. Wilson
intervals are shown beside every rate and decide nothing. A physical error
rate reported `undetermined` cannot serve as a sub-threshold point for a gate.

Decision: `sub-threshold-per-operation-decoder-p-pooled-two-proportion`.
Code: `src/evaluation/distance_scaling.py`, `src/evaluation/stats.py`
(`two_proportion_test`). Every artifact produced under these rules restates
them in its `rules` block.

No earlier result was produced under different rules for either comparison, so
nothing is re-evaluated.

### 9.2 Correction of 2026-10-11 — baselines named as built

§6 named BP+OSD via CUDA-Q as the secondary baseline. No result was ever
produced with BP+OSD: the harness has always decoded Belief-Matching, and §6 now
names it, with every baseline's package, exact pin and configuration. Tesseract's
configuration was fixed (undecomposed DEM, the package's long-beam preset)
before any Tesseract result existed. No decision rule of §1–§5 changes, and no
result is re-evaluated.

Decisions: one per baseline in the Trurlic `evaluation` component, and
`baseline-packages-are-required-exact-pins-imported-at-the`.
