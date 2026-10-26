"""
evaluate_external.py — Genuine external / forward-looking validation.

WHY THIS EXISTS

Every other evaluation in this repo (evaluate_original.py's grouped CV,
stratified CV, temporal split) resamples or re-splits the SAME 638-trial
corpus. That is legitimate for estimating out-of-sample performance, but it
is still one dataset, one snapshot, one label-construction pipeline applied
throughout. A reviewer will reasonably ask: does anything here hold up on
trials that were not part of any development decision at all — not used to
pick features, not used to fit hierarchical priors, not used to tune the
composite weights, not looked at until the very last step?

This script builds that lockbox: it reserves the most recent slice of
trials by start_year (a genuine forward-looking holdout — trials the
"model" would not yet have existed to see, at the time the rest of the
corpus was used for development) and evaluates ONCE, after everything else
(hierarchical rates, composite v2 weights, the genomic-priority model) has
been fit on the earlier slice only. Unlike the internal CV protocols, this
lockbox is used for evaluation exactly one time in this script — if you
re-run this after changing the model based on what you saw here, you have
turned it back into a dev set, not external validation.

USAGE
    python evaluate_external.py                  # default: last 3 start_years locked
    python evaluate_external.py --lockbox-years 2
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from config import DATA_DIR, REPORT_DIR, RANDOM_SEED
from hierarchical_model import fit_hierarchical_rates, lookup_or_backoff
from decision_support import GENOMIC_FEATURES

LABEL = "label_permissive"
rng = np.random.default_rng(RANDOM_SEED)


def bootstrap_ci(y, p, n_boot=1000):
    if len(set(y)) < 2:
        return float("nan"), float("nan")
    n = len(y)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        draws.append(roc_auc_score(y[idx], p[idx]))
    if not draws:
        return float("nan"), float("nan")
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lockbox-years", type=int, default=3,
                    help="number of most-recent distinct start_years to lock box")
    args = ap.parse_args()

    df = pd.read_csv(DATA_DIR / "classification_matrix.csv")
    df = df.loc[:, ~df.columns.duplicated()]
    det = df[df[LABEL] != -1].copy()
    det = det[det.start_year.notna()]

    years = sorted(det.start_year.unique())
    lock_years = set(years[-args.lockbox_years:])
    dev = det[~det.start_year.isin(lock_years)]
    lockbox = det[det.start_year.isin(lock_years)]

    print(f"Development set: {len(dev)} trials, years up to {max(dev.start_year):.0f}")
    print(f"LOCKBOX (never used for fitting): {len(lockbox)} trials, "
          f"years {sorted(int(y) for y in lock_years)}")
    if len(lockbox) < 20:
        print(f"\n  ⚠ Lockbox has only {len(lockbox)} trials — widen "
              f"--lockbox-years, or treat this result as indicative only.")
    if lockbox[LABEL].nunique() < 2:
        print("\n  Lockbox has only one class present — cannot compute AUC. "
              "Widen --lockbox-years.")
        return

    # ── Fit everything on dev only ────────────────────────────────────────────
    cols = [c for c in GENOMIC_FEATURES if c in dev.columns and dev[c].notna().any()]
    X_dev = dev[cols].fillna(dev[cols].median())
    y_dev = dev[LABEL].values
    groups = (dev.primary_target.astype(str) + "|" + dev.disease.astype(str)).values

    base = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(n_estimators=150, max_depth=2,
                                           learning_rate=0.05,
                                           random_state=RANDOM_SEED)),
    ])
    n_splits = min(5, max(2, len(set(groups)) // 4))
    cv = list(GroupKFold(n_splits=n_splits).split(X_dev, y_dev, groups))
    model = CalibratedClassifierCV(base, method="isotonic", cv=cv)
    model.fit(X_dev, y_dev)

    rates = fit_hierarchical_rates(dev, LABEL)

    # ── Evaluate on the untouched lockbox, once ───────────────────────────────
    med = dev[cols].median()
    X_lock = lockbox[cols].fillna(med)
    y_lock = lockbox[LABEL].values
    ml_probs = model.predict_proba(X_lock)[:, 1]

    blended = []
    for (_, row), ml_p in zip(lockbox.iterrows(), ml_probs):
        hier = lookup_or_backoff(rates, row.primary_target, row.disease,
                                 row.get("modality", ""))
        vals = row[cols].astype(float)
        coverage = float(((vals.notna()) & (vals != 0)).mean())
        w = hier["own_data_weight"] * coverage
        blended.append(w * ml_p + (1 - w) * hier["posterior_mean"])
    blended = np.array(blended)

    print(f"\n{'='*70}")
    print(f"  LOCKBOX RESULTS (n={len(lockbox)}, evaluated exactly once)")
    print(f"{'='*70}")
    for name, p in [("ML model alone", ml_probs), ("Blended (ML + hierarchical)", blended)]:
        auc = roc_auc_score(y_lock, p)
        lo, hi = bootstrap_ci(y_lock, p)
        pr = average_precision_score(y_lock, p)
        brier = brier_score_loss(y_lock, p)
        print(f"\n  {name}:")
        print(f"    ROC AUC:  {auc:.3f}  [{lo:.3f}, {hi:.3f}]")
        print(f"    PR AUC:   {pr:.3f}")
        print(f"    Brier:    {brier:.3f}")

    prevalence_auc = 0.5
    print(f"\n  Null (prevalence-only) reference AUC: {prevalence_auc:.3f}")
    print(f"  Lockbox positive rate: {y_lock.mean():.1%} (n_pos={int(y_lock.sum())})")

    pd.DataFrame({
        "nct_id": lockbox.nct_id, "primary_target": lockbox.primary_target,
        "disease": lockbox.disease, "start_year": lockbox.start_year,
        "label": y_lock, "ml_probability": ml_probs, "blended_score": blended,
    }).to_csv(REPORT_DIR / "external_lockbox_results.csv", index=False)
    print(f"\n  -> {REPORT_DIR / 'external_lockbox_results.csv'}")
    print("\n  Do not iterate on the model based on this result and re-run —")
    print("  that converts the lockbox into a dev set. Widen the lockbox or")
    print("  collect a truly separate corpus for further external checks.")


if __name__ == "__main__":
    main()
