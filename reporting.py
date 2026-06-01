"""
reporting.py - All reports required by the task spec.

Produces:
  A. Global cohort report (CSV + printed)
  B. Asset-level report (CSV with per-trial predictions + score breakdown)
  C. Error analysis report (FP/FN analysis)
  D. Sensitivity analysis (subsets, label defs, feature sets)
  E. Data dictionary (describes every column)
  F. Parquet export
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, f1_score, brier_score_loss,
    confusion_matrix, balanced_accuracy_score
)
from sklearn.model_selection import StratifiedKFold

from config import (
    DATA_DIR, REPORT_DIR, RAW_FEATURES, COMPOSITE_FEATURES,
    LABEL_NAMES, RANDOM_SEED, log
)


# ============================================================
# A. GLOBAL COHORT REPORT
# ============================================================
def generate_cohort_report(trials_df, model_df, results_df):
    """Save a structured cohort summary."""
    report = {
        "metric": [],
        "value": [],
    }
    def add(metric, value):
        report["metric"].append(metric)
        report["value"].append(value)

    add("trials_ingested", len(trials_df))
    add("phase2_with_drug", len(trials_df))
    add("labelable", len(model_df))
    add("drug_mapped", int((model_df["canonical_drug"] != "unmapped").sum()))
    add("drug_mapped_pct", round((model_df["canonical_drug"] != "unmapped").mean() * 100, 1))
    add("with_biomarker", int(model_df["biomarker"].notna().sum()))
    add("unique_drugs", model_df["canonical_drug"].nunique())
    add("unique_targets", model_df["primary_target"].nunique())
    add("unique_diseases", model_df["disease"].nunique())

    # Label balance
    for lbl in LABEL_NAMES:
        add(f"{lbl}_positive", int(model_df[lbl].sum()))
        add(f"{lbl}_rate", round(model_df[lbl].mean(), 3))

    # Missingness
    for col in RAW_FEATURES:
        if col in model_df.columns:
            miss = model_df[col].isna().sum()
            if miss > 0:
                add(f"missing_{col}", miss)

    # Real data coverage
    if "has_real_ot_data" in model_df.columns:
        add("real_opentargets_trials", int(model_df["has_real_ot_data"].sum()))
        add("real_pubmed_trials", int(model_df["has_real_pubmed_data"].sum()))

    # Best models
    for lbl in LABEL_NAMES:
        sub = results_df[(results_df["label"] == lbl) & results_df["roc_auc"].notna()]
        if len(sub) > 0:
            best = sub.loc[sub["roc_auc"].idxmax()]
            add(f"best_model_{lbl}", f"{best['model']}_{best['features']}")
            add(f"best_auc_{lbl}", round(best["roc_auc"], 3))
            add(f"best_brier_{lbl}", round(best["brier"], 3))

    report_df = pd.DataFrame(report)
    report_df.to_csv(REPORT_DIR / "cohort_report.csv", index=False)
    print("\n  === COHORT REPORT ===")
    print(report_df.to_string(index=False))
    return report_df


# ============================================================
# B. ASSET-LEVEL REPORT
# ============================================================
def generate_asset_report(data, trained_models, eval_label="label_balanced"):
    """Per-trial report with predictions, score breakdown, strengths/risks."""
    test_df = data["test_df"]
    X_test = data["feature_sets"]["hybrid"][1]

    best_key = f"lgb_hybrid_{eval_label}"
    if best_key not in trained_models:
        best_key = f"lr_hybrid_{eval_label}"
    if best_key not in trained_models:
        print("  No model for asset report")
        return None

    model = trained_models[best_key]
    probs = model.predict_proba(X_test)[:, 1]

    asset = test_df[["nct_id", "canonical_drug", "primary_target", "disease",
                      "biomarker", "modality", eval_label]].copy()
    asset["predicted_p"] = probs.round(4)

    # Add composite scores
    for score in COMPOSITE_FEATURES:
        if score in test_df.columns:
            asset[score] = test_df[score].values

    # Top strength and top risk per trial
    strengths = []
    risks = []
    for _, row in asset.iterrows():
        scores = {s: row.get(s, 0) for s in COMPOSITE_FEATURES if s in row.index and s != "BIOLOGY_SCORE"}
        if scores:
            strengths.append(max(scores, key=scores.get))
            risks.append(min(scores, key=scores.get))
        else:
            strengths.append("N/A")
            risks.append("N/A")
    asset["top_strength"] = strengths
    asset["top_risk"] = risks

    asset = asset.sort_values("predicted_p", ascending=False)
    asset.to_csv(REPORT_DIR / "asset_report.csv", index=False)

    print(f"\n  === ASSET REPORT ({len(asset)} trials) ===")
    print(f"  Top 5 predicted successes:")
    print(asset.head(5)[["nct_id", "canonical_drug", "disease", "predicted_p",
                          "top_strength", "top_risk"]].to_string(index=False))
    print(f"\n  Bottom 5 predicted failures:")
    print(asset.tail(5)[["nct_id", "canonical_drug", "disease", "predicted_p",
                          "top_strength", "top_risk"]].to_string(index=False))
    return asset


# ============================================================
# C. ERROR ANALYSIS
# ============================================================
def generate_error_analysis(data, trained_models, eval_label="label_balanced"):
    """Analyze false positives and false negatives."""
    test_df = data["test_df"]
    X_test = data["feature_sets"]["hybrid"][1]

    best_key = f"lgb_hybrid_{eval_label}"
    if best_key not in trained_models:
        best_key = f"lr_hybrid_{eval_label}"
    if best_key not in trained_models:
        return

    model = trained_models[best_key]
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    y_true = test_df[eval_label].values

    show_cols = ["nct_id", "canonical_drug", "primary_target", "disease", "modality"]

    # False positives
    fp_mask = (preds == 1) & (y_true == 0)
    fp_df = test_df.loc[fp_mask, show_cols].copy()
    fp_df["predicted_p"] = probs[fp_mask]
    fp_df = fp_df.sort_values("predicted_p", ascending=False)

    # False negatives
    fn_mask = (preds == 0) & (y_true == 1)
    fn_df = test_df.loc[fn_mask, show_cols].copy()
    fn_df["predicted_p"] = probs[fn_mask]
    fn_df = fn_df.sort_values("predicted_p", ascending=True)

    # Analyze patterns
    print(f"\n  === ERROR ANALYSIS ===")
    print(f"  False Positives: {len(fp_df)} (predicted success, actually failed)")
    if len(fp_df) > 0:
        print(f"    Most common targets: {fp_df['primary_target'].value_counts().head(3).to_dict()}")
        print(f"    Most common diseases: {fp_df['disease'].value_counts().head(3).to_dict()}")
        print(f"    Top 5 FPs:")
        print(fp_df.head(5).to_string(index=False))

    print(f"\n  False Negatives: {len(fn_df)} (predicted failure, actually succeeded)")
    if len(fn_df) > 0:
        print(f"    Most common targets: {fn_df['primary_target'].value_counts().head(3).to_dict()}")
        print(f"    Most common diseases: {fn_df['disease'].value_counts().head(3).to_dict()}")
        print(f"    Top 5 FNs:")
        print(fn_df.head(5).to_string(index=False))

    # Save
    error_df = pd.concat([
        fp_df.assign(error_type="false_positive"),
        fn_df.assign(error_type="false_negative"),
    ])
    error_df.to_csv(REPORT_DIR / "error_analysis.csv", index=False)
    return error_df


# ============================================================
# D. SENSITIVITY ANALYSIS
# ============================================================
def run_sensitivity_analysis(model_df, data):
    """
    Run sensitivity analyses across:
      - label definitions
      - feature sets
      - subsets (mapped drugs, biomarker enriched, high confidence)
    """
    print(f"\n  === SENSITIVITY ANALYSIS ===")

    results = []

    def quick_train(X_tr, y_tr, X_te, y_te, name):
        if y_tr.sum() < 5 or y_te.sum() < 2:
            return
        pw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
        mdl = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  scale_pos_weight=pw, random_state=RANDOM_SEED, verbose=-1)
        mdl.fit(X_tr, y_tr)
        prob = mdl.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)
        try: auc = roc_auc_score(y_te, prob)
        except: auc = np.nan
        brier = brier_score_loss(y_te, prob)
        f1 = f1_score(y_te, pred, zero_division=0)
        r = {"analysis": name, "auc": round(auc, 3), "brier": round(brier, 3),
             "f1": round(f1, 3), "n_train": len(y_tr), "n_test": len(y_te),
             "pos_rate_train": round(y_tr.mean(), 3)}
        results.append(r)
        print(f"    {name:50s} AUC={auc:.3f} Brier={brier:.3f} F1={f1:.3f}")

    X_train_hybrid = data["feature_sets"]["hybrid"][0]
    X_test_hybrid = data["feature_sets"]["hybrid"][1]
    train_df = data["train_df"]
    test_df = data["test_df"]

    # 1. Label definitions
    print("\n  1. Label definition sensitivity:")
    for lbl in LABEL_NAMES:
        if train_df[lbl].sum() >= 5:
            quick_train(X_train_hybrid, train_df[lbl].values,
                       X_test_hybrid, test_df[lbl].values, f"label={lbl}")

    # 2. Feature sets
    print("\n  2. Feature set sensitivity:")
    for fname, (Xtr, Xte) in data["feature_sets"].items():
        quick_train(Xtr, train_df["label_balanced"].values,
                   Xte, test_df["label_balanced"].values, f"features={fname}")

    # 3. Mapped drugs only
    print("\n  3. Subset sensitivity:")
    mapped_tr = train_df[train_df["canonical_drug"] != "unmapped"]
    mapped_te = test_df[test_df["canonical_drug"] != "unmapped"]
    if len(mapped_tr) > 20 and len(mapped_te) > 10:
        quick_train(X_train_hybrid.loc[mapped_tr.index], mapped_tr["label_balanced"].values,
                   X_test_hybrid.loc[mapped_te.index], mapped_te["label_balanced"].values,
                   f"mapped_drugs_only (n={len(mapped_tr)}+{len(mapped_te)})")

    # 4. Biomarker enriched
    bmk_tr = train_df[train_df["biomarker"].notna()]
    bmk_te = test_df[test_df["biomarker"].notna()]
    if len(bmk_tr) > 20 and len(bmk_te) > 10:
        quick_train(X_train_hybrid.loc[bmk_tr.index], bmk_tr["label_balanced"].values,
                   X_test_hybrid.loc[bmk_te.index], bmk_te["label_balanced"].values,
                   f"biomarker_enriched (n={len(bmk_tr)}+{len(bmk_te)})")

    # 5. High confidence labels
    hc_tr = train_df[train_df["label_confidence"] >= 0.60]
    hc_te = test_df[test_df["label_confidence"] >= 0.60]
    if len(hc_tr) > 20 and len(hc_te) > 10:
        quick_train(X_train_hybrid.loc[hc_tr.index], hc_tr["label_balanced"].values,
                   X_test_hybrid.loc[hc_te.index], hc_te["label_balanced"].values,
                   f"high_confidence_labels (n={len(hc_tr)}+{len(hc_te)})")

    # 6. Targeted therapy only (exclude chemo)
    chemo_targets = {"DNA", "TUBB", "TYMS", "RRM1", "TOP1", "TOP2A", "DHFR"}
    targeted_tr = train_df[~train_df["primary_target"].isin(chemo_targets | {"UNKNOWN"})]
    targeted_te = test_df[~test_df["primary_target"].isin(chemo_targets | {"UNKNOWN"})]
    if len(targeted_tr) > 20 and len(targeted_te) > 10:
        quick_train(X_train_hybrid.loc[targeted_tr.index], targeted_tr["label_balanced"].values,
                   X_test_hybrid.loc[targeted_te.index], targeted_te["label_balanced"].values,
                   f"targeted_therapy_only (n={len(targeted_tr)}+{len(targeted_te)})")

    sens_df = pd.DataFrame(results)
    sens_df.to_csv(REPORT_DIR / "sensitivity_analysis.csv", index=False)
    return sens_df


# ============================================================
# E. CROSS-VALIDATION
# ============================================================
def run_cross_validation(model_df, n_folds=5):
    """Run stratified k-fold cross-validation and report fold variability."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    print(f"\n  === {n_folds}-FOLD CROSS-VALIDATION ===")
    label = "label_balanced"
    y = model_df[label].values

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    features = [c for c in RAW_FEATURES + COMPOSITE_FEATURES if c in model_df.columns]
    X_raw = model_df[features].copy()
    X_imp = pd.DataFrame(imputer.fit_transform(X_raw), columns=features, index=model_df.index)
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=features, index=model_df.index)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_tr, X_te = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        pw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
        mdl = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  scale_pos_weight=pw, random_state=RANDOM_SEED, verbose=-1)
        mdl.fit(X_tr, y_tr)
        prob = mdl.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)

        try: auc = roc_auc_score(y_te, prob)
        except: auc = np.nan
        brier = brier_score_loss(y_te, prob)
        f1 = f1_score(y_te, pred, zero_division=0)
        ba = balanced_accuracy_score(y_te, pred)

        fold_results.append({"fold": fold+1, "auc": auc, "brier": brier, "f1": f1, "balanced_acc": ba})
        print(f"    Fold {fold+1}: AUC={auc:.3f} Brier={brier:.3f} F1={f1:.3f}")

    cv_df = pd.DataFrame(fold_results)
    print(f"\n    Mean AUC: {cv_df['auc'].mean():.3f} ± {cv_df['auc'].std():.3f}")
    print(f"    Mean Brier: {cv_df['brier'].mean():.3f} ± {cv_df['brier'].std():.3f}")
    print(f"    Mean F1: {cv_df['f1'].mean():.3f} ± {cv_df['f1'].std():.3f}")

    cv_df.to_csv(REPORT_DIR / "cross_validation.csv", index=False)
    return cv_df


# ============================================================
# F. DATA DICTIONARY + PARQUET EXPORT
# ============================================================
def export_data_dictionary(model_df):
    """Generate a data dictionary describing every column."""
    descriptions = {
        "nct_id": "ClinicalTrials.gov trial identifier",
        "canonical_drug": "Normalized drug name from curated dictionary",
        "primary_target": "Primary gene target (HGNC symbol)",
        "disease": "Normalized cancer type",
        "biomarker": "Biomarker extracted from trial text (if any)",
        "modality": "Drug modality: small_molecule, antibody, adc, etc.",
        "moa": "Mechanism of action category",
        "drug_confidence": "Confidence in drug normalization (0-1)",
        "disease_confidence": "Confidence in disease normalization (0-1)",
        "status": "ClinicalTrials.gov trial status",
        "start_year": "Year trial started",
        "sponsor": "Lead sponsor organization",
        # Raw features A
        "target_disease_assoc": "Open Targets overall association score (real or curated)",
        "genetics_evidence": "Genetic association evidence score",
        "somatic_evidence": "Somatic mutation evidence score",
        "pathway_evidence": "Pathway-level evidence score",
        "known_drug_evidence": "Known drug evidence score",
        "literature_diversity": "Publication diversity (log-scaled count)",
        # Raw features B
        "dependency_mean": "Mean DepMap dependency score in relevant lineage",
        "dependency_selectivity": "Biomarker+ vs biomarker- selectivity",
        "dependency_variance": "Dependency variance across cell lines",
        "context_specificity": "Context specificity index",
        # Raw features C
        "degree_centrality": "STRING network degree centrality (normalized)",
        "network_confidence": "Network interaction confidence",
        "clustering_coefficient": "Network clustering coefficient",
        "pathway_redundancy": "Pathway redundancy proxy",
        "bottleneck_score": "Network bottleneck/chokepoint proxy",
        "alternative_pathway_burden": "Alternative pathway burden",
        # Raw features D
        "alteration_frequency": "Biomarker alteration frequency in disease",
        "biomarker_directness": "Biomarker-target directness score",
        "co_alteration_burden": "Co-alteration burden",
        "genomic_complexity": "Genomic context complexity",
        "assay_precision": "Biomarker assay precision proxy",
        "biomarker_prevalence": "Biomarker prevalence in disease",
        "biomarker_clonality": "Biomarker clonality proxy",
        "biomarker_missing": "1 if no biomarker identified, 0 otherwise",
        # Raw features E
        "modality_target_fit": "Modality-target fit score",
        "binding_evidence": "Known target-binding evidence",
        "potency_proxy": "Drug potency proxy",
        "functional_annotation_depth": "UniProt functional annotation depth",
        "domain_criticality": "Target domain criticality",
        "perturbation_concordance": "Perturbation concordance score",
        "mechanistic_maturity": "Overall mechanistic maturity",
        # Raw features F
        "tumor_expression": "Tumor-relevant expression proxy",
        "normal_tissue_burden": "Normal tissue expression burden",
        "expression_variability": "Expression variability across tissues",
        "tumor_specificity": "Tumor specificity proxy",
        "therapeutic_window": "Therapeutic window proxy",
        # Raw features G
        "pub_velocity_1y": "Publication velocity (1-year)",
        "pub_acceleration": "Publication acceleration",
        "pub_concentration": "Publication concentration index",
        "mechanistic_study_fraction": "Fraction of mechanistic studies",
        "translational_study_fraction": "Fraction of translational studies",
        "human_study_fraction": "Fraction of human/clinical studies",
        # Raw features H
        "has_known_ligands": "1 if known ligands exist, 0 otherwise",
        "tractability": "Target tractability proxy",
        "polypharmacology_proxy": "Polypharmacology risk proxy",
        # Data source flags
        "has_real_ot_data": "1 if real Open Targets API data used, 0 if curated fallback",
        "has_real_pubmed_data": "1 if real PubMed API data used, 0 if curated fallback",
        # Composite scores
        "GCS": "Genetic Causality Score",
        "DS": "Dependency Score",
        "SS": "Selectivity Score",
        "CSI": "Context Specificity Index",
        "NCS": "Network Centrality Score",
        "PRS": "Pathway Redundancy Score",
        "BS": "Bottleneck Score",
        "FAS": "Functional Annotation Score",
        "TSS_exp": "Tumor Specificity Score (expression)",
        "SWP": "Safety Window Proxy",
        "PCS": "Perturbation Concordance Score",
        "AFS": "Alteration Frequency Score",
        "CABS": "Co-alteration Burden Score",
        "MMS": "Mechanistic Maturity Score",
        "PMS": "Publication Momentum Score",
        "CDS": "Causal Dependency Score (higher-order)",
        "TDS": "Target Dominance Score (higher-order)",
        "BFS": "Biomarker Fidelity Score (higher-order)",
        "MCS": "Mechanistic Coherence Score (higher-order)",
        "TWS": "Therapeutic Window Score (higher-order)",
        "EMS": "Evidence Momentum Score (higher-order)",
        "BIOLOGY_SCORE": "Overall biology score (weighted combination of CDS/TDS/BFS/MCS/TWS/EMS)",
        # Labels
        "label_strict": "Strict label: 1=clearly positive results only",
        "label_balanced": "Balanced label: 1=positive results or strong biology completion (~33%)",
        "label_permissive": "Permissive label: 1=any completion or continuation signal (~74%)",
        "label_provenance": "How this label was assigned (human-readable reason)",
        "label_confidence": "Confidence in label assignment (0-1)",
    }

    dict_rows = []
    for col in model_df.columns:
        dict_rows.append({
            "column": col,
            "description": descriptions.get(col, ""),
            "dtype": str(model_df[col].dtype),
            "non_null": int(model_df[col].notna().sum()),
            "missing": int(model_df[col].isna().sum()),
            "example": str(model_df[col].dropna().iloc[0]) if model_df[col].notna().any() else "N/A",
        })

    dict_df = pd.DataFrame(dict_rows)
    dict_df.to_csv(DATA_DIR / "data_dictionary.csv", index=False)
    print(f"\n  Data dictionary saved: {len(dict_df)} columns documented")

    # Parquet export
    try:
        parquet_cols = [c for c in model_df.columns if model_df[c].dtype != object or c in
                       ["nct_id", "canonical_drug", "primary_target", "disease", "biomarker",
                        "modality", "moa", "status", "sponsor", "label_provenance"]]
        model_df[parquet_cols].to_parquet(DATA_DIR / "classification_matrix.parquet", index=False)
        print(f"  Parquet export saved: classification_matrix.parquet")
    except Exception as e:
        print(f"  Parquet export skipped: {e}")

    return dict_df