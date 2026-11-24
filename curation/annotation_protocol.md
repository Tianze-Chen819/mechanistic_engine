# Outcome annotation protocol

Owner: Arabella Chua · Status: working draft, also the basis for the curation Methods section

## Purpose

Automated labels come from keyword rules on CT.gov records (`label_provenance`). The largest
rule, `completed+results+no_positive_signal` (479 of 638 trials, all labelled 0), treats the
absence of positive wording as failure. Manual annotation replaces these labels with
trial-specific evidence and measures how often the rules are wrong.

## Unit and label

- Unit: one Phase 2 interventional oncology trial (NCT ID), start year 2005 or later.
- Label (`outcome_label`): **1** = met its own pre-specified primary objective; **0** = did not;
  **U** = cannot be determined from available evidence; **EXCLUDE** = out of scope.
- Exclusions: Phase 1/2 or dose-finding; primary endpoint is safety only; withdrawn; start before 2005.

## Selecting trials

- Batches are drawn with `curation_tools.py queue`: stratified random sampling by
  `label_provenance`, fixed seed, excluding trials already reviewed. Each batch's strata sizes
  are logged in `sampling_log.csv`, so per-rule error rates can be re-weighted to the full cohort.
- Selection never uses model predictions, and automated labels are not shown during review.

## Evidence hierarchy

1. Posted CT.gov primary-outcome results with a pre-specified analysis.
2. Peer-reviewed publication of the trial (matched by NCT number), using the statistical
   design and primary result, not the conclusion's tone.
3. Protocol / statistical analysis plan on CT.gov (pre-specified null and target rates).
4. Termination reason (futility → 0; accrual, funding or business → U).
5. If none of these gives a pre-specified criterion and its result → **U**.

## Recording

For every trial: label, source (`ctgov_results`, `publication:<PMID>`, `protocol_sap`,
`termination_text`), one quoted evidence sentence, evidence date (for historical
availability), confidence (high/medium/low), and the reason for any U or exclusion.

## Quality control

- A random 20% of trials are reviewed independently by a second reviewer (Tianze).
  Report percent agreement and Cohen's kappa (`curation_tools.py gold --review2`).
- Disagreements are adjudicated by discussion; the original labels and the adjudicated
  label are both kept.
- Label changes are traceable (`reviewer`, `review_date`, `git` history) and made without
  reference to model scores.

## Outputs used in the paper

- `label_agreement_auto_vs_manual.csv`: agreement between rules and manual review, by rule.
- `cohort_with_manual_labels.csv`: cohort with manual labels attached; automated label
  columns are never overwritten.
- Counts of U and EXCLUDE with reasons (reported, not hidden).

## Open decision for Prof. Rao

About half of reviewed trials are U because no pre-specified success criterion is
published. Options: (a) keep the strict definition and accept fewer labels; (b) add a
secondary label based on the authors' stated conclusion, used only in sensitivity analyses.
