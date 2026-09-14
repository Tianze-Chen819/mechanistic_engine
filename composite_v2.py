"""
composite_v2.py — Data-driven refit of the composite biology scores.

WHY THIS EXISTS

`evaluate_original.py` found that the hand-tuned composite scores (CDS, TDS,
BFS, MCS, TWS, EMS, BIOLOGY_SCORE in features.py) perform WORSE under grouped
cross-validation than simply handing the model the raw features they were
built from (composite best AUC 0.534 vs raw-features best AUC 0.576-0.612).
The weights in features.py (e.g. "cosmic_census_member: 0.25, somatic_evidence:
0.20, ...") were picked by hand, not fit to data — there is no reason to
expect a guessed weighting to beat a model that can weight the same inputs
itself.

This module keeps the exact same idea (six named, biologically-labelled
composite dimensions, each built from the same sub-features features.py
already assigns to it) but LEARNS the weight of each sub-feature from the
labelled data instead of guessing it. Concretely: for each composite, fit an
L2-regularized logistic regression restricted to only that composite's member
features against the outcome label, take the fitted coefficients, clip any
negative coefficient to 0 (a negative weight on a raw feature that was chosen
specifically because it is thought to indicate STRONGER biology would reverse
the intended meaning of the composite — see NOTE below), and renormalize to
sum to 1 so the composite stays on the same 0-1 scale as before.

NOTE ON CLIPPING: a handful of member features may end up with a fitted
negative weight (the audit already flagged pubmed_pair_count and
clinical_trial_pub_count as features whose relationship with the label is not
robust — see docs/label_audit.md). Clipping to zero means "the data does not
support this sub-feature contributing positively to the composite", which is
itself a useful, reportable finding, not just a technical necessity.

Weights should be refit whenever the label set changes materially (e.g. after
each pipeline re-run) — call `fit_composite_v2_weights` fresh rather than
hardcoding the numbers this file prints.

USAGE
    python composite_v2.py     # fits weights, reports v1 vs v2 under grouped CV
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import CDS_WEIGHTS, TDS_WEIGHTS, BFS_WEIGHTS, MCS_WEIGHTS, TWS_WEIGHTS, EMS_WEIGHTS

COMPOSITE_MEMBERS = {
    "CDS": list(CDS_WEIGHTS), "TDS": list(TDS_WEIGHTS), "BFS": list(BFS_WEIGHTS),
    "MCS": list(MCS_WEIGHTS), "TWS": list(TWS_WEIGHTS), "EMS": list(EMS_WEIGHTS),
}


def fit_composite_v2_weights(df: pd.DataFrame, label_col: str = "label_permissive",
                             min_weight_floor: float = 0.05, n_splits: int = 5,
                             n_repeats: int = 20) -> dict:
    """
    Fit data-driven weights for each composite. Returns {composite: {feature: weight}}
    with weights non-negative and summing to 1 per composite.

    Weights are averaged over repeated GroupKFold splits (grouped by
    target+disease, same grouping used everywhere else in this pipeline)
    rather than taken from one fit on the full dataset. A single fit on
    strongly correlated sub-features (e.g. pubmed_pair_count,
    literature_diversity and human_study_fraction are all built from the same
    underlying publication counts) can concentrate almost all the weight onto
    one arbitrary member of a correlated group — averaging many resampled
    fits spreads that weight across the correlated group instead, which is a
    more honest description of "these features carry overlapping signal"
    than picking one and zeroing the rest. C is deliberately small (strong L2
    shrinkage) for the same reason.
    """
    det = df[df[label_col] != -1]
    y_full = det[label_col].values
    weights = {}
    for comp, members in COMPOSITE_MEMBERS.items():
        cols = [c for c in members if c in det.columns
                and det[c].notna().any()]  # drop all-NaN columns (e.g. a raw
                                            # feature source that returned no
                                            # data for this run) before fitting
        if len(cols) < 2:
            weights[comp] = {c: 1 / max(len(cols), 1) for c in cols}
            continue
        X_full = det[cols].values
        groups = (det.primary_target.astype(str) + "|" + det.disease.astype(str)).values

        coef_sum = np.zeros(len(cols))
        n_fits = 0
        for rep in range(n_repeats):
            gkf = GroupKFold(n_splits=n_splits)
            # GroupKFold has no shuffle param; vary the fold assignment across
            # repeats by permuting row order, so different repeats see
            # different group-to-fold assignments
            perm = np.random.RandomState(rep).permutation(len(X_full))
            X, y, g = X_full[perm], y_full[perm], groups[perm]
            for tr, _ in gkf.split(X, y, g):
                if len(set(y[tr])) < 2:
                    continue
                pipe = Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, C=0.15,
                                               class_weight="balanced",
                                               random_state=42)),
                ])
                pipe.fit(X[tr], y[tr])
                coef_sum += np.clip(pipe.named_steps["clf"].coef_[0], 0, None)
                n_fits += 1

        coefs = coef_sum / max(n_fits, 1)
        if coefs.sum() <= 0:
            coefs = np.ones(len(cols))
        # floor so no member is fit to exactly zero and silently disappears —
        # a small floor keeps the composite's original biological scope
        # intact while still reflecting that the feature contributed weakly
        coefs = coefs + min_weight_floor * coefs.sum()
        w = coefs / coefs.sum()
        weights[comp] = dict(zip(cols, w.round(4)))
    return weights


def apply_composite_v2(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Compute v2 composite columns (suffixed _v2) from fitted weights."""
    out = df.copy()
    for comp, w in weights.items():
        cols = [c for c in w if c in out.columns]
        if not cols:
            out[f"{comp}_v2"] = 0.0
            continue
        vals = out[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        out[f"{comp}_v2"] = sum(vals[c] * w[c] for c in cols)
    bio_cols = [f"{c}_v2" for c in COMPOSITE_MEMBERS if f"{c}_v2" in out.columns]
    if bio_cols:
        out["BIOLOGY_SCORE_v2"] = out[bio_cols].mean(axis=1)
    return out


def _grouped_cv_auc(df, cols, label_col, n_splits=5):
    det = df[df[label_col] != -1].dropna(subset=cols)
    if len(det) < 30:
        return float("nan")
    groups = (det.primary_target.astype(str) + "|" + det.disease.astype(str)).values
    y = det[label_col].values
    X = det[cols].values
    gkf = GroupKFold(n_splits=n_splits)
    ys, ps = [], []
    for tr, te in gkf.split(X, y, groups):
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.5,
                                       class_weight="balanced", random_state=42)),
        ])
        pipe.fit(X[tr], y[tr])
        ys.append(y[te])
        ps.append(pipe.predict_proba(X[te])[:, 1])
    y_all, p_all = np.concatenate(ys), np.concatenate(ps)
    if len(set(y_all)) < 2:
        return float("nan")
    return roc_auc_score(y_all, p_all)


if __name__ == "__main__":
    from config import DATA_DIR, REPORT_DIR

    df = pd.read_csv(DATA_DIR / "classification_matrix.csv")
    df = df.loc[:, ~df.columns.duplicated()]
    label_col = "label_permissive"

    weights = fit_composite_v2_weights(df, label_col)
    df2 = apply_composite_v2(df, weights)

    print("=== v2 composite weights (data-fit, non-negative, sum to 1) ===")
    rows = []
    for comp, w in weights.items():
        print(f"\n{comp}:")
        for feat, val in sorted(w.items(), key=lambda kv: -kv[1]):
            old = {"CDS": CDS_WEIGHTS, "TDS": TDS_WEIGHTS, "BFS": BFS_WEIGHTS,
                   "MCS": MCS_WEIGHTS, "TWS": TWS_WEIGHTS, "EMS": EMS_WEIGHTS}[comp].get(feat, 0)
            print(f"  {feat:<28} v1={old:.2f}  v2={val:.3f}")
            rows.append({"composite": comp, "feature": feat, "v1_weight": old, "v2_weight": val})
    pd.DataFrame(rows).to_csv(REPORT_DIR / "composite_v2_weights.csv", index=False)

    print("\n=== v1 vs v2 under grouped 5-fold CV (AUC, GroupKFold on target+disease) ===")
    v1_cols = ["CDS", "TDS", "BFS", "MCS", "TWS", "EMS"]
    v2_cols = [f"{c}_v2" for c in v1_cols]
    v1_auc = _grouped_cv_auc(df, [c for c in v1_cols if c in df.columns], label_col)
    v2_auc = _grouped_cv_auc(df2, [c for c in v2_cols if c in df2.columns], label_col)
    print(f"  composite v1 (hand-tuned):  AUC = {v1_auc:.3f}")
    print(f"  composite v2 (data-fit):    AUC = {v2_auc:.3f}")
    if v2_auc > v1_auc:
        print(f"  -> v2 improves on v1 by {v2_auc - v1_auc:+.3f}")
    else:
        print(f"  -> v2 does not beat v1 ({v2_auc - v1_auc:+.3f}) — report this honestly")
