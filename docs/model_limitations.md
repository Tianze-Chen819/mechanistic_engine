# Fixing the six limitations — what changed and what the evidence shows

Written for: whoever picks this repo up next and wants to know why these
modules exist before touching them.

This documents the six limitations identified after the v9 label rewrite,
what was built to address each one, and — honestly — which fixes are proven
on data already in hand versus which are correctly built but still waiting on
the label-rebuild pipeline to finish before their payoff can be measured.

Run `python audit_labels.py` and `docs/label_audit.md` for the label-side
history this builds on.

---

## 1. Small effective sample size (~200 profiles from 638 rows)

**Fix: `hierarchical_model.py` — empirical-Bayes shrinkage.**

A raw success rate from 1-3 trials is a coin flip dressed up as a statistic.
The fix is the standard one for "too little data to estimate a group's rate
on its own": pool it with similar groups. Trials are grouped into 11
mechanism families (immune-checkpoint, angiogenesis, RAS/MAPK pathway,
cytotoxic chemo, etc. — see `TARGET_FAMILY` in the module), and each
(target, disease) pair's estimate is a Beta-Binomial posterior that blends
its own data with its family's rate. A pair seen once shrinks hard toward
its family; a pair seen 40 times relies almost entirely on itself.

**Proven now.** Example from the current (pre-rebuild) data: `TUBB` in
ovarian cancer has 1 trial, 0 positive — the raw rate is 0%, but the shrunk
estimate is 9.5% [0%, 44%], reflecting the cytotoxic-chemo family's actual
base rate rather than claiming false certainty from n=1. `DNA` in
hepatocellular carcinoma has 1 trial, 1 positive — raw rate 100%, shrunk to
46% [7%, 89%]. This is real and correct regardless of which labels are
loaded, since it operates on whatever determinate labels exist.

This doesn't manufacture more data — it stops the small-n groups from
reporting nonsense confidence, and gives every asset an honest credible
interval instead of a single number.

## 2. Dataset skew toward already-validated biology

**Partial fix: `auto_target_mapping.py` — ChEMBL-sourced drug/target
discovery**, wired into `normalize.py` as a third fallback tier.

The corpus is skewed because the curated `drug_target_db.py` (~100 hand-typed
drugs) only recognizes already-well-known, mostly-approved mechanisms —
anything else falls to "unmapped" and drops out of the biology-scored
dataset entirely. Hand-typing more entries from memory would not safely fix
this (no way to audit where a guessed mapping came from). Instead,
`auto_target_mapping.py` queries ChEMBL's own structured mechanism-of-action
database — drug name → ChEMBL molecule → mechanism record → target → HGNC
gene symbol — and caches every result with its ChEMBL IDs for audit.

**Proven on individual lookups, not yet measured on the corpus.** Verified
against known pharmacology: `capmatinib`→MET, `tepotinib`→MET,
`mobocertinib`→EGFR, `sacituzumab govitecan`→TACSTD2, `repotrectinib`→NTRK1 —
all correct. `placebo` and combination-regimen strings are correctly
rejected before wasting an API call. It's wired into `normalize_drug()` as
the tier tried before giving up (confidence capped at 0.6, tagged
`moa="chembl_auto"` so it can be filtered out for a stricter analysis).

**Caveat:** the currently-running pipeline re-run had already executed
`normalize_trials()` (using the old, curated-only code) before this fix
landed, so it will NOT show up in the matrix this run produces. A subsequent
full pipeline run is needed to measure how much this raises the "% mapped"
figure and whether it meaningfully broadens the target diversity — that
number is not fabricated here because it hasn't been generated yet.

**Honestly unresolved:** even with auto-mapping, the underlying CT.gov corpus
itself is what it is — Phase 2 oncology trials are disproportionately run on
mechanisms sponsors already believe in. No amount of normalization fixes that
selection effect; it can only be reduced (more trials get a real biology
profile) not eliminated. This is worth stating explicitly in any paper.

## 3. Composite scores underperform the raw features they're built from

**Fix: `composite_v2.py` — data-fit weights, same sub-feature membership.**

The v1 composite scores (CDS, TDS, BFS, MCS, TWS, EMS) use hand-picked
weights (e.g. `cosmic_census_member: 0.25`) that were never fit to outcome
data. `composite_v2.py` keeps each composite's biological scope (the same
member features) but learns each member's weight via L2-regularized logistic
regression, averaged over repeated grouped-CV folds (grouped by
target+disease, C=0.15) so the result isn't dominated by one arbitrary
feature in a correlated cluster — an earlier, unaveraged version of this fit
put 93% of EMS's weight on a single feature purely from fold-to-fold noise.

**Measured, and the honest result is: not yet an improvement.** On the
current (pre-rebuild) data, v1 composite AUC = 0.509, v2 = 0.507 under
grouped 5-fold CV — a difference within noise. This is very likely an
artifact of the data these composites are being fit on right now: `BFS`'s
best member features (`alteration_frequency`, `lineage_specificity`) are
almost entirely NaN in the current matrix because of the cBioPortal API bug
fixed earlier this session — the refit has nothing real to learn from for
that composite until the pipeline re-run lands real genomic pair-level data.
**Re-run `composite_v2.py` once the new matrix is in place; the fitted
weights and the v1-vs-v2 comparison should both change materially.**

## 4. Literature/publication features don't carry significant signal

**Not "fixed" — and forcing this to look significant would be p-hacking.**
This was already investigated at length: PubMed queries are now
date-bounded, alias-aware, and log-scaled (see `docs/label_audit.md`), and
the publication-linking pipeline itself had a real bug (silently-failing
JSON parse) that's now fixed (see the p-value-extraction work in
`clients_v8_additions.py`). Those were real defects worth fixing regardless.
But the audit's finding that literature volume doesn't clear the
statistical-significance bar under a fair, grouped evaluation may simply be
correct — publication count is a proxy for how well-known a drug already is,
which is a weaker and more confounded signal than direct genomic evidence.
**Recommendation: report this as a finding, not a bug to keep chasing.**

## 5. Only genomic features show a real, significant signal

**Fix: `decision_support.py` — a genomic-priority production model**, rather
than diluting the working signal by throwing every feature (including the
ones that don't help) into one model.

This directly acts on finding #4/#5 together: instead of an omnibus model,
the production model here is trained specifically on the genomic feature
subset (COSMIC, DepMap, GWAS, somatic evidence, cBioPortal alteration
frequency/lineage specificity), calibrated with isotonic regression fit via
grouped cross-validation (not on the model's own training predictions, which
would overstate confidence).

**A real bug was caught and fixed while building this**: an all-zero feature
vector (an unmapped drug, or a mapped target with literally no genomic-source
data) landed in a gradient-boosting leaf that happened to have the single
HIGHEST fitted probability in the entire dataset (0.632) — meaning "we have
zero genomic evidence about this drug" would have been the model's #1
recommendation. `_genomic_evidence_coverage()` now measures how much of a
row's genomic feature vector is actually populated, and gates the ML
component's weight in the blended score by that coverage — a row with no
evidence gets scored entirely from the hierarchical family base rate, and is
excluded from the headline "top assets" ranking (still visible in the full
CSV, clearly flagged). This is the kind of failure mode that only shows up
when you actually try to use a model's output for ranking, not when you just
look at its aggregate AUC — worth remembering for any future scoring tool.

## 6. Single dataset, single time window, no external validation

**Fix: `evaluate_external.py` — a genuine forward-looking lockbox.**

Reserves the most recent N start-years of trials, fits everything
(hierarchical rates, the genomic-priority model, calibration) on the earlier
slice only, and evaluates on the untouched lockbox exactly once. This is not
a second independent data source (that would require a second registry or
disease area, out of scope here) but it is a real, if modest, external check:
trials the model genuinely could not have seen during development.

**Measured, and the honest result on current data is concerning, not
reassuring**: on a 5-year lockbox (n=36, 8 positives), ROC AUC = 0.306 [0.131,
0.488] for the ML model alone, 0.319 [0.135, 0.509] blended — both **below**
the 0.5 null, though the confidence interval is wide enough to include chance
performance given only 8 positive lockbox trials. This is not proof the model
is anti-predictive; it is a small, noisy sample. But it is real evidence that
the modest positive signal seen under grouped CV on historical data does not
yet demonstrably carry forward to genuinely unseen, more recent trials — and
that is exactly the kind of result external validation is supposed to be
able to show. **Do not re-tune the model against this specific lockbox
result and re-run it** — that defeats the purpose; widen the lockbox or wait
for more recent trials to accrue enough positives for a stable estimate.

---

## What to re-run once the label-rebuild pipeline finishes

The v9 label rewrite (real trial outcomes instead of pre-trial text) and the
cBioPortal/PubMed repairs are running as a full pipeline re-run in the
background as of this writing. Once `classification_matrix.csv` reflects
that run:

```bash
python hierarchical_model.py       # re-fit shrinkage rates on the new labels
python composite_v2.py             # re-fit composite weights — BFS/CDS should
                                    # change the most, now that cBioPortal
                                    # genomic features are populated
python decision_support.py --top 10
python evaluate_external.py --lockbox-years 3
python evaluate_original.py        # updated headline AUC/CI numbers
```

None of the numbers quoted above (composite v1 vs v2, the 0.306 lockbox AUC)
should be treated as final — they were measured against the matrix available
before the rebuild, specifically so each module's mechanics could be verified
against real data rather than shipped untested. Re-running them is the last
step, not optional.
