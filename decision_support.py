"""
decision_support.py — Turn biology signals into an actual decision-support
output, instead of a single opaque probability.

WHY THIS EXISTS

A bare "P(success) = 0.63" from a model trained on ~200 effective biology
profiles is false precision — it invites a reader to treat two candidates
scoring 0.61 and 0.65 as meaningfully different, when the honest uncertainty
on either number spans most of the 0-1 range. This module builds the report
a decision-maker can actually use responsibly:

  1. GENOMIC-PRIORITY MODEL: evaluate_original.py's grouped-CV evaluation
     found the only feature group with a statistically significant grouped-CV
     signal is the genomic one (COSMIC/DepMap/GWAS/somatic/cBioPortal
     mutation data) — literature and hand-built composite scores did not
     clear the null baseline. So the production model here is trained on
     that feature set specifically, rather than everything at once, which
     dilutes the working signal with non-working features.
  2. HIERARCHICAL RATE (hierarchical_model.py): a shrinkage-based base rate
     for the same (target, disease) pair, with an honest credible interval
     that widens for sparse pairs instead of overstating confidence.
  3. BLENDED SCORE: the two are combined, weighted by how much data actually
     supports the ML model's view of this specific pair (own_data_weight from
     the hierarchical fit) — for a pair the model has barely seen, the
     report leans on the hierarchical base rate; for a well-populated pair,
     it leans on the discriminative model.
  4. PERCENTILE, NOT RAW PROBABILITY: the headline number reported is this
     asset's percentile rank among all historical biology profiles, which is
     a fairer summary of what a biology-only score can actually tell you
     ("this profile resembles the top quartile of historical profiles that
     succeeded") than a probability that implies more precision than a ~200
     effective-sample model supports.
  5. ACTIONABLE LEVERS: reuses counterfactual.py's composite-to-raw-feature
     mapping to report which specific, named biological weaknesses would
     most change the assessment if addressed (e.g. biomarker enrichment),
     tagged by clinical feasibility.
  6. EXPLICIT CAVEATS on every report — this is a biology-only, small-sample
     estimate, not a go/no-go verdict.

This is decision SUPPORT, explicitly not decision automation — see the
caveats block on every generated report.

USAGE
    python decision_support.py                    # full cohort report
    python decision_support.py --nct NCT01234567   # one asset's report
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from config import DATA_DIR, REPORT_DIR, RANDOM_SEED, ALL_RAW_FEATURES
from hierarchical_model import fit_hierarchical_rates, lookup_or_backoff
from counterfactual import COMPOSITE_TO_RAW, SCORE_FEASIBILITY

LABEL = "label_permissive"

GENOMIC_FEATURES = [c for c in ALL_RAW_FEATURES if any(
    k in c for k in ("alteration", "lineage", "genomic", "cosmic", "somatic",
                     "depmap", "gwas"))]


def _load_matrix() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "classification_matrix.csv")
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def train_genomic_priority_model(df: pd.DataFrame):
    """
    Calibrated model on the genomic feature set — see module docstring for why
    this feature set, not the full raw set or the composite scores, is the
    production choice.

    Calibration uses isotonic regression via CalibratedClassifierCV with
    GroupKFold (grouped by target+disease) so calibration is fit on held-out
    predictions, not the model's own training-fold outputs — an uncalibrated
    or self-calibrated model would report overconfident probabilities on the
    exact profiles it was fit on.
    """
    det = df[df[LABEL] != -1].copy()
    cols = [c for c in GENOMIC_FEATURES if c in det.columns and det[c].notna().any()]
    X = det[cols].fillna(det[cols].median())
    y = det[LABEL].values
    groups = (det.primary_target.astype(str) + "|" + det.disease.astype(str)).values

    base = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(n_estimators=150, max_depth=2,
                                           learning_rate=0.05,
                                           random_state=RANDOM_SEED)),
    ])
    n_groups = len(set(groups))
    n_splits = min(5, max(2, n_groups // 4))
    cv = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    model = CalibratedClassifierCV(base, method="isotonic", cv=cv)
    model.fit(X, y)
    return model, cols, det


def percentile_rank(scores: pd.Series, value: float) -> float:
    return float((scores < value).mean() * 100)


def actionable_levers(row: pd.Series, model, cols: list[str], top_n: int = 3) -> list[dict]:
    """
    Reuse counterfactual.py's composite->raw mapping restricted to the
    genomic-priority model's own feature set, so the reported levers are ones
    this specific model actually responds to.
    """
    baseline = model.predict_proba(row[cols].to_frame().T)[:, 1][0]
    levers = []
    for score_name, raw_feats in COMPOSITE_TO_RAW.items():
        feats_in_model = [f for f in raw_feats if f in cols]
        if not feats_in_model:
            continue
        perturbed = row.copy()
        for f in feats_in_model:
            cur = float(row[f]) if pd.notna(row[f]) and row[f] != -1 else 0.0
            perturbed[f] = min(cur + 0.15, 1.0)
        new_p = model.predict_proba(perturbed[cols].to_frame().T)[:, 1][0]
        levers.append({
            "lever": score_name, "delta_p": round(new_p - baseline, 4),
            "feasibility": SCORE_FEASIBILITY.get(score_name, "UNKNOWN"),
        })
    levers.sort(key=lambda d: -d["delta_p"])
    return levers[:top_n]


def _genomic_evidence_coverage(row: pd.Series, cols: list[str]) -> float:
    """
    Fraction of genomic-priority features that carry real information for
    this row (non-null and non-zero). An UNMAPPED target (or one with zero
    coverage from every genomic source) has an all-zero feature vector by
    construction — that is an ABSENCE of evidence, not evidence of low risk.

    This matters because a tree-based model can map an all-zero input to a
    leaf with a high fitted probability purely from how few (and by chance,
    which) training rows happened to land there — a genuinely observed test
    case here mapped 45 unmapped-target trials to the SINGLE HIGHEST
    probability in the entire dataset (0.632), which would have made "we
    don't know anything about this drug's target" the model's top-ranked
    recommendation. Coverage gates how much the ML component is trusted.
    """
    vals = row[cols]
    informative = vals.notna() & (vals != 0)
    return float(informative.mean())


def build_asset_report(row: pd.Series, model, cols: list[str],
                       rates: pd.DataFrame, all_probs: pd.Series) -> dict:
    target, disease = row["primary_target"], row["disease"]
    modality = row.get("modality", "")

    ml_prob = float(model.predict_proba(row[cols].to_frame().T)[:, 1][0])
    hier = lookup_or_backoff(rates, target, disease, modality)
    coverage = _genomic_evidence_coverage(row, cols)

    # Blend weighted by (a) how much this pair's OWN data supports the
    # hierarchical view, AND (b) how much genomic evidence backs the ML
    # model's view of this row at all. A row with zero genomic feature
    # coverage (target == UNKNOWN, or a mapped target with no genomic-source
    # data) gets its ML weight zeroed regardless of own_data_weight — an
    # unsupported model output should never outrank a real evidence-based
    # base rate. See _genomic_evidence_coverage.
    w = hier["own_data_weight"] * coverage
    blended = w * ml_prob + (1 - w) * hier["posterior_mean"]

    pct = percentile_rank(all_probs, ml_prob)
    levers = actionable_levers(row, model, cols) if coverage > 0 else []

    return {
        "nct_id": row.get("nct_id", ""),
        "canonical_drug": row.get("canonical_drug", ""),
        "primary_target": target, "disease": disease,
        "mechanism_family": hier["family"],
        "genomic_evidence_coverage": round(coverage, 2),
        "ml_model_probability": round(ml_prob, 3),
        "hierarchical_base_rate": round(hier["posterior_mean"], 3),
        "hierarchical_ci": f"[{hier['ci_lo']:.2f}, {hier['ci_hi']:.2f}]",
        "pair_n_trials_seen": hier["n"],
        "own_data_weight": round(w, 2),
        "blended_decision_score": round(blended, 3),
        "percentile_vs_historical_profiles": round(pct, 1),
        "top_levers": levers,
        "caveats": _caveats(hier, coverage) + (
            [_lever_caveat(levers)] if _lever_caveat(levers) else []),
    }


def _caveats(hier: dict, coverage: float) -> list[str]:
    notes = [
        "Biology-only estimate: excludes protocol design, site execution, "
        "dosing, and patient-selection quality by design — see README.",
        "Not a go/no-go verdict. Treat as one structured input among several.",
    ]
    if coverage == 0:
        notes.append(
            "NO genomic evidence for this target (unmapped drug, or no "
            "source covers it) — the ML component is fully discounted; this "
            "score is the mechanism family's historical base rate only, not "
            "an assessment of this target's actual biology.")
    elif coverage < 0.3:
        notes.append(
            f"Sparse genomic evidence ({coverage:.0%} of features populated) "
            f"— the ML component is down-weighted accordingly.")
    if hier["n"] == 0:
        notes.append(
            f"No prior trials of this exact (target, disease) pair in the "
            f"training corpus — this estimate rests entirely on the "
            f"'{hier['family']}' mechanism family's historical rate.")
    elif hier["own_data_weight"] < 0.3:
        notes.append(
            f"Only {hier['n']} prior trial(s) of this exact pair — the "
            f"blended score leans heavily on the family base rate, not this "
            f"pair's own (small, noisy) history.")
    if hier["ci_hi"] - hier["ci_lo"] > 0.5:
        notes.append("Wide credible interval — treat the point estimate as "
                     "indicative only.")
    return notes


def _lever_caveat(levers: list[dict]) -> str | None:
    if levers and all(abs(lv["delta_p"]) < 0.005 for lv in levers):
        return ("No lever in the genomic-priority feature set moves this "
               "specific profile's score — its relevant features are likely "
               "already at a plateau (e.g. a categorical evidence flag "
               "already at its maximum value) rather than genuinely "
               "unimprovable; interpret 'no actionable lever' accordingly.")
    return None


def print_report(report: dict):
    print(f"\n{'='*70}")
    print(f"  {report['nct_id']}  |  {report['canonical_drug']}  |  "
          f"{report['primary_target']} in {report['disease']}")
    print(f"{'='*70}")
    print(f"  Mechanism family:         {report['mechanism_family']}")
    print(f"  ML model probability:     {report['ml_model_probability']:.3f}  "
          f"(genomic-priority model)")
    print(f"  Hierarchical base rate:   {report['hierarchical_base_rate']:.3f}  "
          f"90% CI {report['hierarchical_ci']}  "
          f"(from {report['pair_n_trials_seen']} prior trial(s) of this pair)")
    print(f"  --> Blended decision score: {report['blended_decision_score']:.3f}  "
          f"(own-data weight {report['own_data_weight']:.2f})")
    print(f"  Percentile vs. historical biology profiles: "
          f"{report['percentile_vs_historical_profiles']:.0f}th")
    print(f"\n  Top actionable levers:")
    for lv in report["top_levers"]:
        print(f"    {lv['lever']:<6} Δp={lv['delta_p']:+.3f}   {lv['feasibility']}")
    print(f"\n  Caveats:")
    for c in report["caveats"]:
        print(f"    - {c}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nct", default=None, help="report a single trial by NCT id")
    ap.add_argument("--top", type=int, default=10,
                    help="cohort mode: show this many highest-percentile assets")
    args = ap.parse_args()

    df = _load_matrix()
    model, cols, det = train_genomic_priority_model(df)
    rates = fit_hierarchical_rates(df, LABEL)
    all_probs = pd.Series(model.predict_proba(det[cols].fillna(det[cols].median()))[:, 1],
                          index=det.index)

    if args.nct:
        row = df[df.nct_id == args.nct]
        if row.empty:
            print(f"NCT id {args.nct} not found in matrix")
            return
        row = row.iloc[0].copy()
        for c in cols:
            if pd.isna(row.get(c)):
                row[c] = det[c].median()
        report = build_asset_report(row, model, cols, rates, all_probs)
        print_report(report)
        return

    print(f"Genomic-priority model trained on {len(cols)} features, "
          f"{len(det)} determinate trials")
    print(f"Feature set: {cols}\n")

    reports = []
    for idx in det.index[:200]:  # cap for runtime; full run via --nct for one asset
        reports.append(build_asset_report(det.loc[idx], model, cols, rates, all_probs))
    rep_df = pd.DataFrame(reports)
    rep_df.drop(columns=["top_levers", "caveats"]).to_csv(
        REPORT_DIR / "decision_support_report.csv", index=False)

    # Zero-coverage rows (unmapped target, or no genomic source covers it)
    # are excluded from the headline ranking — see _genomic_evidence_coverage.
    # Their rows are still in the saved CSV, flagged, just not presented as
    # top picks.
    ranked = rep_df[rep_df.genomic_evidence_coverage > 0].sort_values(
        "blended_decision_score", ascending=False)
    n_excluded = (rep_df.genomic_evidence_coverage == 0).sum()

    print(f"=== Top {args.top} assets by blended decision score "
          f"(genomic evidence required) ===")
    for _, r in ranked.head(args.top).iterrows():
        print(f"  {r.nct_id:<14} {r.canonical_drug:<20} {r.primary_target:<8} "
              f"{r.disease:<20} blended={r.blended_decision_score:.3f}  "
              f"pct={r.percentile_vs_historical_profiles:>5.1f}  "
              f"n_seen={r.pair_n_trials_seen}")
    if n_excluded:
        print(f"\n  ({n_excluded} assets excluded from ranking — zero genomic "
              f"evidence coverage, see decision_support_report.csv)")
    print(f"\nFull cohort report -> {REPORT_DIR / 'decision_support_report.csv'}")
    print("\nFor a single asset's full report with actionable levers:")
    print("  python decision_support.py --nct <NCT_ID>")


if __name__ == "__main__":
    main()
