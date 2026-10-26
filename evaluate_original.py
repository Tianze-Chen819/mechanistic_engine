"""
evaluate_original.py — Honest evaluation of the original (real-API) data.

The pipeline's own evaluation reports a single AUC per (model, feature set,
label) cell against a temporal split. That number is not interpretable on its
own here, for three reasons this script addresses directly:

  * n=198 test rows with 38 positives. The 95% CI on an AUC that small spans
    roughly +/-0.09, so 0.63 and 0.55 are not distinguishable.
  * The temporal split is not grouped. 43% of test rows share a (drug, disease)
    pair with a training row, and the biology features are a pure function of
    (target, disease) — so those rows are near-duplicates.
  * There is no null baseline. Without one you cannot tell whether an AUC of
    0.63 reflects biology or the protocol-verbosity confound.

So every model here is scored under four protocols and against three nulls,
with bootstrap intervals, and a permutation test for the headline number.

Run:
    python evaluate_original.py                    # all protocols
    python evaluate_original.py --protocol grouped
    python evaluate_original.py --n-boot 2000
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_DIR, REPORT_DIR, RANDOM_SEED, TEST_YEAR_CUTOFF,
    ALL_RAW_FEATURES, COMPOSITE_SCORES,
)

warnings.filterwarnings("ignore")
rng = np.random.default_rng(RANDOM_SEED)

LABEL = "label_permissive"


# ── models ────────────────────────────────────────────────────────────────────

def get_models():
    return {
        "LogReg": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.1,
                                       class_weight="balanced",
                                       random_state=RANDOM_SEED)),
        ]),
        "RandomForest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=4, min_samples_leaf=20,
                class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
        "GradBoost": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=2, learning_rate=0.05,
                random_state=RANDOM_SEED)),
        ]),
    }


# ── metrics with uncertainty ──────────────────────────────────────────────────

def bootstrap_auc(y, p, n_boot=1000):
    """Percentile bootstrap CI. Resamples rows, skipping degenerate draws."""
    y, p = np.asarray(y), np.asarray(p)
    if len(set(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    point = roc_auc_score(y, p)
    draws = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        draws.append(roc_auc_score(y[idx], p[idx]))
    if not draws:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def permutation_p(y, p, n_perm=1000):
    """P(AUC >= observed | labels shuffled). The honest 'is this real' test."""
    y, p = np.asarray(y), np.asarray(p)
    if len(set(y)) < 2:
        return float("nan")
    obs = roc_auc_score(y, p)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = roc_auc_score(rng.permutation(y), p)
    return float((np.sum(null >= obs) + 1) / (n_perm + 1))


def score(y, p, n_boot):
    auc, lo, hi = bootstrap_auc(y, p, n_boot)
    return {
        "roc_auc": auc, "auc_lo": lo, "auc_hi": hi,
        "pr_auc": average_precision_score(y, p) if len(set(y)) > 1 else float("nan"),
        "brier": brier_score_loss(y, p),
        "n": len(y), "n_pos": int(np.sum(y)),
        "prevalence": float(np.mean(y)),
    }


# ── evaluation protocols ──────────────────────────────────────────────────────

def protocol_temporal_unpurged(df):
    """
    Temporal split with NO group purge — the pre-v9 pipeline behaviour, kept
    here only to show how much a temporal-only split overstates performance.
    run_pipeline.py's temporal_split() defaults to group_purge=True now;
    this protocol reproduces what it did before that fix.
    """
    tr = df.start_year <= TEST_YEAR_CUTOFF
    yield "temporal_unpurged", df[tr], df[~tr]


def protocol_temporal(df):
    """
    What the pipeline does today: temporal split, then drop test rows whose
    (target, disease) pair also appears in training (run_pipeline.py's
    temporal_split(..., group_purge=True), now the default). This is the
    split that actually asks 'does this generalise to biology the model has
    not seen?', using the same year cutoff CT.gov trial dates are compared
    against elsewhere in the pipeline.
    """
    tr = df[df.start_year <= TEST_YEAR_CUTOFF]
    te = df[df.start_year > TEST_YEAR_CUTOFF]
    seen = set(zip(tr.primary_target, tr.disease))
    mask = ~te.apply(lambda r: (r.primary_target, r.disease) in seen, axis=1)
    yield "temporal_grouped", tr, te[mask]


def protocol_grouped_cv(df, n_splits=5):
    """
    GroupKFold on (target, disease). Every fold's test biology is absent from
    its training set, and all rows are used, so the estimate is far less noisy
    than a single 198-row holdout.
    """
    groups = (df.primary_target.astype(str) + "|" + df.disease.astype(str)).values
    gkf = GroupKFold(n_splits=n_splits)
    for i, (tr, te) in enumerate(gkf.split(df, df[LABEL], groups)):
        yield f"grouped_cv_fold{i}", df.iloc[tr], df.iloc[te]


def protocol_stratified_cv(df, n_splits=5):
    """Plain stratified CV — deliberately optimistic, included as an upper bound."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    for i, (tr, te) in enumerate(skf.split(df, df[LABEL])):
        yield f"stratified_cv_fold{i}", df.iloc[tr], df.iloc[te]


# ── null baselines ────────────────────────────────────────────────────────────

def null_baselines(df, feature_cols):
    """
    Three references the biology model must beat to mean anything.

      prevalence    predict the base rate for everyone -> AUC 0.5 by construction
      n_features    how many non-missing biology features the row has. Pure
                    data-availability, zero mechanism. Well-studied targets have
                    more populated features AND more literature.
      missingness   the missing-data indicator columns alone.
    """
    out = {}
    y = df[LABEL].values

    out["prevalence"] = np.full(len(df), df[LABEL].mean())

    bio = df[[c for c in feature_cols if not c.endswith("_missing")]]
    out["data_availability"] = bio.notna().sum(axis=1).values.astype(float)

    miss_cols = [c for c in df.columns if c.endswith("_missing")]
    if miss_cols:
        out["missingness_only"] = df[miss_cols].fillna(1).sum(axis=1).values.astype(float)

    return out, y


# ── feature sets and ablations ────────────────────────────────────────────────

def feature_sets(df):
    raw = [c for c in ALL_RAW_FEATURES if c in df.columns]
    comp = [c for c in COMPOSITE_SCORES if c in df.columns]
    lit = [c for c in raw if any(k in c for k in
                                 ("pubmed", "literature", "pub_", "known_drug",
                                  "clinical_trial_pub", "interaction_score",
                                  "ot_known_drug", "ot_literature"))]
    genomic = [c for c in raw if any(k in c for k in
                                     ("alteration", "lineage", "genomic", "cosmic",
                                      "somatic", "depmap", "gwas"))]
    return {
        "raw": raw,
        "composite": comp,
        "raw_no_literature": [c for c in raw if c not in lit],
        "literature_only": lit,
        "genomic_only": genomic,
    }


# ── main loop ─────────────────────────────────────────────────────────────────

def run(df, protocols, fsets, n_boot, n_perm):
    rows = []
    models = get_models()

    for fs_name, cols in fsets.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        for proto_name, proto_fn in protocols.items():
            # CV protocols yield many folds; pool out-of-fold predictions so the
            # AUC is computed once on all n rows rather than averaged over folds.
            pooled = {m: ([], []) for m in models}
            single = None
            for split_name, tr, te in proto_fn(df):
                if len(te) < 10 or tr[LABEL].nunique() < 2 or te[LABEL].nunique() < 2:
                    continue
                Xtr, ytr = tr[cols], tr[LABEL].values
                Xte, yte = te[cols], te[LABEL].values
                for mname, model in models.items():
                    m = get_models()[mname]
                    m.fit(Xtr, ytr)
                    p = m.predict_proba(Xte)[:, 1]
                    pooled[mname][0].append(yte)
                    pooled[mname][1].append(p)
                single = split_name

            for mname, (ys, ps) in pooled.items():
                if not ys:
                    continue
                y = np.concatenate(ys)
                p = np.concatenate(ps)
                s = score(y, p, n_boot)
                s.update(protocol=proto_name, features=fs_name, model=mname,
                         n_features=len(cols))
                if proto_name in ("temporal", "grouped_cv"):
                    s["perm_p"] = permutation_p(y, p, n_perm)
                rows.append(s)
                print(f"  {proto_name:<22} {fs_name:<20} {mname:<14} "
                      f"AUC={s['roc_auc']:.3f} [{s['auc_lo']:.3f},{s['auc_hi']:.3f}] "
                      f"PR={s['pr_auc']:.3f} n={s['n']}"
                      + (f" p={s['perm_p']:.3f}" if "perm_p" in s else ""))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--protocol", default="all",
                    choices=["all", "temporal", "grouped", "unpurged", "stratified"])
    args = ap.parse_args()

    path = DATA_DIR / "classification_matrix.csv"
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.duplicated()]          # matrix ships dup columns
    df = df[df[LABEL] != -1].copy()
    df = df[df.start_year.notna()]
    print(f"loaded {len(df)} determinate trials  "
          f"({df[LABEL].mean():.1%} positive, {df[LABEL].sum():.0f} positives)")

    fsets = feature_sets(df)
    all_cols = fsets["raw"] + fsets["composite"]

    # ── nulls first: these set the bar ────────────────────────────────────────
    print("\n" + "=" * 78)
    print("NULL BASELINES — what you get with no biology at all")
    print("=" * 78)
    nulls, y = null_baselines(df, all_cols)
    null_rows = []
    for name, p in nulls.items():
        if len(set(p)) < 2:
            print(f"  {name:<22} AUC=0.500 (constant by construction)")
            null_rows.append(dict(protocol="baseline", features=name, model="baseline",
                                  roc_auc=0.5, auc_lo=np.nan, auc_hi=np.nan,
                                  n=len(y), n_pos=int(y.sum())))
            continue
        auc, lo, hi = bootstrap_auc(y, p, args.n_boot)
        pp = permutation_p(y, p, args.n_perm)
        print(f"  {name:<22} AUC={auc:.3f} [{lo:.3f},{hi:.3f}]  perm p={pp:.3f}")
        null_rows.append(dict(protocol="baseline", features=name, model="baseline",
                              roc_auc=auc, auc_lo=lo, auc_hi=hi, perm_p=pp,
                              n=len(y), n_pos=int(y.sum())))

    # ── protocols ─────────────────────────────────────────────────────────────
    available = {
        "temporal": protocol_temporal,             # pipeline default (grouped)
        "unpurged": protocol_temporal_unpurged,     # pre-v9 behaviour, for comparison
        "grouped_cv": protocol_grouped_cv,
        "stratified_cv": protocol_stratified_cv,
    }
    if args.protocol != "all":
        key = {"grouped": "grouped_cv", "stratified": "stratified_cv"}.get(
            args.protocol, args.protocol)
        available = {key: available[key]}

    print("\n" + "=" * 78)
    print("MODELS")
    print("=" * 78)
    res = run(df, available, fsets, args.n_boot, args.n_perm)
    res = pd.concat([pd.DataFrame(null_rows), res], ignore_index=True)

    dest = REPORT_DIR / "evaluation_original.csv"
    res.to_csv(dest, index=False)

    # ── interpretation ────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("READ-OUT")
    print("=" * 78)
    best_null = max((r["roc_auc"] for r in null_rows if not np.isnan(r["roc_auc"])),
                    default=0.5)
    print(f"  strongest null baseline:      AUC {best_null:.3f}")

    models_only = res[res.protocol != "baseline"]
    for proto in models_only.protocol.unique():
        sub = models_only[models_only.protocol == proto]
        b = sub.loc[sub.roc_auc.idxmax()]
        beats = "yes" if b.auc_lo > best_null else "NO"
        print(f"  {proto:<22} best AUC {b.roc_auc:.3f} "
              f"[{b.auc_lo:.3f},{b.auc_hi:.3f}] ({b.model}/{b.features})"
              f"  CI clears null: {beats}")

    lit = models_only[models_only.features == "literature_only"]
    nolit = models_only[models_only.features == "raw_no_literature"]
    if len(lit) and len(nolit):
        print(f"\n  literature features alone:    AUC {lit.roc_auc.max():.3f}")
        print(f"  biology minus literature:     AUC {nolit.roc_auc.max():.3f}")
        print("  If these are close, the model is reading publication volume,")
        print("  not mechanism — and publication volume is a proxy for how famous")
        print("  the drug already is.")

    print(f"\n  -> {dest}")


if __name__ == "__main__":
    main()
