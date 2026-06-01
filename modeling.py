"""
modeling.py — Model training, evaluation, and analysis.

v6 CHANGES:
  - Stronger regularization on all models (CV AUC was 0.463 vs test 0.570 → overfitting)
  - LightGBM: lower learning rate, fewer leaves, stronger L1/L2 reg, min_child_samples
  - RandomForest: fewer estimators, higher min_samples_leaf
  - Added SHAP analysis per label definition
  - Pristine analysis now compares full vs. characterized-only datasets
  - Feature importance split by pair-level vs target-level features
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    brier_score_loss, confusion_matrix, balanced_accuracy_score,
    RocCurveDisplay, PrecisionRecallDisplay,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("  LightGBM not available — using ExtraTreesClassifier as substitute")
    from sklearn.ensemble import ExtraTreesClassifier

from config import RANDOM_SEED, CV_FOLDS, FIG_DIR, REPORT_DIR, MODEL_DIR, log

# ── Model definitions with stronger regularization ────────────────────────────

def get_models():
    """
    v7 MODEL IMPROVEMENTS:
    - Test set is now larger (cutoff 2015 vs 2018), so we can use slightly
      less regularization without as much overfitting risk
    - Added GradientBoosting as an additional model type
    - Scale-aware LogReg with StandardScaler pipeline
    - LightGBM tuned for class imbalance (scale_pos_weight)
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    models = {}

    # Logistic Regression — plain (Pipeline causes multi_class conflicts in sklearn)
    models["LogReg"] = LogisticRegression(
        C=0.1, max_iter=2000, random_state=RANDOM_SEED,
        class_weight="balanced", solver="lbfgs",
    )

    # Calibrated LogReg
    base_lr = LogisticRegression(
        C=0.1, max_iter=2000, random_state=RANDOM_SEED,
        class_weight="balanced", solver="lbfgs",
    )
    models["LogReg_Cal"] = CalibratedClassifierCV(base_lr, cv=3, method="isotonic")

    # RandomForest — slightly relaxed regularization with larger test set
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=8,
        min_samples_split=15,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
    )

    # Gradient Boosting — often better than RF on small datasets
    models["GradBoost"] = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=RANDOM_SEED,
    )

    # LightGBM — tuned for 17% positive rate imbalance
    if HAS_LGB:
        # scale_pos_weight = neg/pos ratio to handle imbalance
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=5,
            min_child_samples=15,
            reg_alpha=0.3,
            reg_lambda=0.3,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=4.0,  # ~515/109 ≈ 4.7, use 4.0 to avoid over-correction
            random_state=RANDOM_SEED,
            verbose=-1,
        )
    else:
        models["LightGBM"] = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            random_state=RANDOM_SEED,
        )

    return models

# ── Metrics ───────────────────────────────────────────────────────────────────

def _safe_f1(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return f1_score(y_true, preds, zero_division=0)

def evaluate(model, X_test, y_test, label_name="", model_name="", feature_set=""):
    probs = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
    prauc = average_precision_score(y_test, probs)
    f1    = _safe_f1(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    bacc  = balanced_accuracy_score(y_test, (probs >= 0.5).astype(int))
    return {
        "label": label_name, "model": model_name, "features": feature_set,
        "roc_auc": round(auc, 6), "pr_auc": round(prauc, 6),
        "f1": round(f1, 6), "brier": round(brier, 6),
        "balanced_accuracy": round(bacc, 6),
    }

# ── Training ──────────────────────────────────────────────────────────────────

def train_all_models(data: dict) -> dict:
    """
    Train all model configurations.
    data: output of build_modeling_data()
    Returns: {label_key: {model_key: fitted_model}}
    """
    label_defs = ["label_strict", "label_balanced", "label_permissive"]
    # v7: permissive is primary — more training data, still circularity-free
    feature_sets = data["feature_sets"]
    trained = {}
    all_results = []

    for label_name in label_defs:
        trained[label_name] = {}
        y_train = data["train_labels"][label_name]
        y_test  = data["test_labels"][label_name]

        print(f"\n  --- {label_name} ---")
        label_results = []

        for fs_name, (X_train, X_test) in feature_sets.items():
            models = get_models()
            for model_name, model in models.items():
                try:
                    # Filter out indeterminate (-1) labels before fitting
                    train_mask = y_train.isin([0, 1])
                    test_mask  = y_test.isin([0, 1])
                    Xtr = X_train[train_mask]
                    ytr = y_train[train_mask]
                    Xte = X_test[test_mask]
                    yte = y_test[test_mask]
                    if len(ytr) < 10 or len(np.unique(ytr)) < 2:
                        continue
                    if len(yte) < 5 or len(np.unique(yte)) < 2:
                        continue
                    model.fit(Xtr, ytr)
                    row = evaluate(model, Xte, yte, label_name, model_name, fs_name)
                    label_results.append(row)
                    key = f"{label_name}__{model_name}__{fs_name}"
                    trained[label_name][key] = model
                except Exception as e:
                    log.warning(f"Training failed: {label_name}/{model_name}/{fs_name}: {e}")

        if not label_results:
            print(f"  WARNING: All models failed for {label_name}")
            continue
        df_lr = pd.DataFrame(label_results).sort_values("roc_auc", ascending=False)
        print(df_lr[["label", "model", "features", "roc_auc", "pr_auc", "f1", "brier", "balanced_accuracy"]].to_string(index=False))
        all_results.extend(label_results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    # Best models
    print("\n  === BEST MODELS ===")
    best = {}
    for label_name in label_defs:
        subset = results_df[results_df["label"] == label_name]
        if subset.empty:
            continue
        best_row = subset.sort_values("roc_auc", ascending=False).iloc[0]
        key = f"{label_name}__{best_row['model']}__{best_row['features']}"
        best[label_name] = {
            "key": key, "model": trained[label_name].get(key),
            "model_name": best_row["model"], "features": best_row["features"],
            "auc": best_row["roc_auc"], "brier": best_row["brier"],
        }
        print(f"    {label_name:<22} -> {best_row['model']} ({best_row['features']}) AUC={best_row['roc_auc']:.3f} Brier={best_row['brier']:.3f}")

    log.info(f"Trained {len(all_results)} model configurations")
    return {"trained": trained, "best": best, "results": results_df}

# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate(X: pd.DataFrame, y: pd.Series, n_folds: int = CV_FOLDS, label_name: str = "label_permissive") -> dict:
    """5-fold stratified cross-validation on the best model type (LightGBM composite)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    aucs, briers, f1s = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        # Filter indeterminate labels
        tr_mask = y_tr.isin([0, 1])
        val_mask = y_val.isin([0, 1])
        X_tr, y_tr = X_tr[tr_mask], y_tr[tr_mask]
        X_val, y_val = X_val[val_mask], y_val[val_mask]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
            continue
        # Deduplicate columns to prevent LightGBM crash
        X_tr = X_tr.loc[:, ~X_tr.columns.duplicated()]
        X_val = X_val.loc[:, ~X_val.columns.duplicated()]
        models = get_models()
        model = models["LightGBM"]
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, probs) if len(np.unique(y_val)) > 1 else 0.5
        brier = brier_score_loss(y_val, probs)
        f1 = _safe_f1(y_val, probs)
        aucs.append(auc)
        briers.append(brier)
        f1s.append(f1)
        print(f"    Fold {fold+1}: AUC={auc:.3f} Brier={brier:.3f} F1={f1:.3f}")

    result = {
        "mean_auc": np.mean(aucs), "std_auc": np.std(aucs),
        "mean_brier": np.mean(briers), "std_brier": np.std(briers),
        "mean_f1": np.mean(f1s), "std_f1": np.std(f1s),
    }
    print(f"\n    Mean AUC: {result['mean_auc']:.3f} ± {result['std_auc']:.3f}")
    print(f"    Mean Brier: {result['mean_brier']:.3f} ± {result['std_brier']:.3f}")
    print(f"    Mean F1: {result['mean_f1']:.3f} ± {result['std_f1']:.3f}")
    return result

# ── Evaluation plots ──────────────────────────────────────────────────────────

def plot_evaluation(best_models: dict, data: dict, label_name: str = "label_balanced"):
    """Plot ROC, PR, calibration, and confusion matrices."""
    best = best_models.get(label_name)
    if not best or not best.get("model"):
        return

    model = best["model"]
    fs_name = best["features"]
    X_test, y_test = data["feature_sets"][fs_name][1], data["test_labels"][label_name]

    # Filter indeterminate labels for evaluation
    clean_mask = y_test.isin([0, 1])
    X_test = X_test[clean_mask]
    y_test = y_test[clean_mask]
    if len(y_test) == 0 or len(np.unique(y_test)) < 2:
        log.warning("Insufficient clean labels for evaluation plots")
        return
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    RocCurveDisplay.from_predictions(y_test, probs, ax=axes[0], name=best["model_name"])
    axes[0].set_title(f"ROC Curve ({label_name})")
    axes[0].plot([0,1],[0,1],"k--",alpha=0.5)

    PrecisionRecallDisplay.from_predictions(y_test, probs, ax=axes[1], name=best["model_name"])
    axes[1].set_title(f"PR Curve ({label_name})")

    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
    axes[2].plot(mean_pred, frac_pos, "s-", label=best["model_name"])
    axes[2].plot([0,1],[0,1],"k--",alpha=0.5,label="Perfect")
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Fraction positive")
    axes[2].set_title(f"Calibration ({label_name})")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "evaluation_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved evaluation_curves.png")

    # Confusion matrices
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, preds)
    im = ax2.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax2)
    ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
    ax2.set_xticklabels(["Pred Neg","Pred Pos"]); ax2.set_yticklabels(["Actual Neg","Actual Pos"])
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, cm[i,j], ha="center", va="center",
                     color="white" if cm[i,j] > cm.max()/2 else "black")
    ax2.set_title(f"Confusion Matrix ({label_name})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved confusion_matrices.png")

# ── Feature importance ────────────────────────────────────────────────────────

PAIR_LEVEL_COLS = {
    "ot_overall_score", "ot_genetic_association", "ot_somatic_mutation",
    "ot_known_drug", "ot_animal_model", "ot_rna_expression", "ot_literature",
    "pubmed_pair_count", "clinical_trial_pub_count", "pair_pub_acceleration",
    "lineage_specificity", "gwas_disease_specificity", "alteration_frequency",
    "co_alteration_burden", "genomic_complexity",
}

def plot_feature_importance(best_models: dict, data: dict, label_name: str = "label_balanced"):
    """Plot feature importance with pair-level vs target-level distinction."""
    best = best_models.get(label_name)
    if not best or not best.get("model"):
        return

    model = best["model"]
    fs_name = best["features"]
    X_train = data["feature_sets"][fs_name][0]

    # Get importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_names = X_train.columns.tolist()
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        feat_names = X_train.columns.tolist()
    else:
        return

    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
    imp_df["is_pair_level"] = imp_df["feature"].isin(PAIR_LEVEL_COLS)
    imp_df = imp_df.sort_values("importance", ascending=False)

    print(f"\n  Top 15 features ({label_name}):")
    print(imp_df.head(15)[["feature", "importance", "is_pair_level"]].to_string(index=False))

    pair_total = imp_df[imp_df["is_pair_level"]]["importance"].sum()
    tgt_total = imp_df[~imp_df["is_pair_level"]]["importance"].sum()
    total = pair_total + tgt_total
    print(f"\n  Pair-level feature importance: {pair_total/total:.1%}")
    print(f"  Target-level feature importance: {tgt_total/total:.1%}")

    fig, ax = plt.subplots(figsize=(10, 6))
    top15 = imp_df.head(15)
    colors = ["#E8593C" if is_pair else "#3B8BD4"
              for is_pair in top15["is_pair_level"]]
    ax.barh(top15["feature"][::-1], top15["importance"][::-1], color=colors[::-1])
    ax.set_title(f"Feature Importance ({label_name})\nOrange=pair-level, Blue=target-level")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved feature_importance.png")

    # SHAP
    try:
        X_sample = data["feature_sets"][fs_name][1].sample(min(100, len(data["feature_sets"][fs_name][1])),
                                                             random_state=RANDOM_SEED)
        if hasattr(model, "predict_proba"):
            explainer = shap.TreeExplainer(model) if hasattr(model, "feature_importances_") \
                        else shap.LinearExplainer(model, X_sample)
            shap_vals = explainer.shap_values(X_sample)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            fig_s, ax_s = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_vals, X_sample, show=False, max_display=15)
            plt.tight_layout()
            plt.savefig(FIG_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("  SHAP saved")
    except Exception as e:
        log.warning(f"SHAP failed: {e}")

    return imp_df

# ── Sensitivity analysis ──────────────────────────────────────────────────────

def sensitivity_analysis(data: dict) -> pd.DataFrame:
    """
    Test model performance across different conditions.
    v6 addition: tests pair-level vs target-level feature performance.
    """
    results = []
    label_name = "label_balanced"
    y_train = data["train_labels"][label_name]
    y_test = data["test_labels"][label_name]

    print(f"\n  === SENSITIVITY ANALYSIS ===")

    # 1. Label definition sensitivity
    print(f"\n  1. Label definition sensitivity:")
    for ln in ["label_strict", "label_balanced", "label_permissive"]:
        yt = data["test_labels"].get(ln)
        ytrain = data["train_labels"].get(ln)
        if yt is None or ytrain is None:
            continue
        X_tr, X_te = data["feature_sets"]["composite"]
        models = get_models()
        model = models["LightGBM"]
        try:
            tr_mask = ytrain.isin([0, 1])
            te_mask = yt.isin([0, 1])
            model.fit(X_tr[tr_mask], ytrain[tr_mask])
            row = evaluate(model, X_te[te_mask], yt[te_mask], ln, "LightGBM", "composite")
            print(f"    label={ln:<30} AUC={row['roc_auc']:.3f} Brier={row['brier']:.3f} F1={row['f1']:.3f}")
            results.append({"test": "label_sensitivity", **row})
        except Exception as e:
            log.warning(f"Sensitivity failed: {e}")

    # 2. Feature set sensitivity
    print(f"\n  2. Feature set sensitivity:")
    for fs_name in ["raw", "composite", "hybrid"]:
        if fs_name not in data["feature_sets"]:
            continue
        X_tr, X_te = data["feature_sets"][fs_name]
        models = get_models()
        model = models["LightGBM"]
        try:
            tr_m = y_train.isin([0, 1])
            te_m = y_test.isin([0, 1])
            model.fit(X_tr[tr_m], y_train[tr_m])
            row = evaluate(model, X_te[te_m], y_test[te_m], label_name, "LightGBM", fs_name)
            print(f"    features={fs_name:<20} AUC={row['roc_auc']:.3f} Brier={row['brier']:.3f} F1={row['f1']:.3f}")
            results.append({"test": "feature_sensitivity", **row})
        except Exception as e:
            log.warning(f"Sensitivity failed: {e}")

    # 3. Subset sensitivity
    print(f"\n  3. Subset sensitivity:")
    train_df = data["train_df"]
    test_df = data["test_df"]
    X_tr_composite, X_te_composite = data["feature_sets"]["composite"]

    subsets = {
        "mapped_drugs_only":     test_df["primary_target"] != "UNKNOWN",
        "biomarker_enriched":    test_df["biomarker_missing"] == 0
                                  if "biomarker_missing" in test_df.columns
                                  else pd.Series([True] * len(test_df), index=test_df.index),
        "has_real_ot_data":      test_df["ot_data_missing"] == 0
                                  if "ot_data_missing" in test_df.columns
                                  else pd.Series([False] * len(test_df), index=test_df.index),
    }

    for subset_name, mask in subsets.items():
        X_sub = X_te_composite[mask]
        y_sub = y_test[mask]
        if len(y_sub) < 20 or len(np.unique(y_sub)) < 2:
            continue
        models = get_models()
        model = models["LightGBM"]
        try:
            # Filter -1 labels from both train and subset
            tr_clean = y_train.isin([0, 1])
            sub_clean = y_sub.isin([0, 1])
            y_sub_f = y_sub[sub_clean]
            X_sub_f = X_sub[sub_clean]
            if len(y_sub_f) < 10 or len(np.unique(y_sub_f)) < 2:
                continue
            model.fit(X_tr_composite[tr_clean], y_train[tr_clean])
            row = evaluate(model, X_sub_f, y_sub_f, label_name, "LightGBM", "composite")
            n_sub = len(y_sub_f)
            print(f"    {subset_name:<30} (n={n_sub}) AUC={row['roc_auc']:.3f} Brier={row['brier']:.3f} F1={row['f1']:.3f}")
            results.append({"test": f"subset_{subset_name}", "n": n_sub, **row})
        except Exception as e:
            log.warning(f"Subset analysis failed for {subset_name}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(REPORT_DIR / "sensitivity_analysis.csv", index=False)
    return df

# ── Pristine analysis: characterized vs. well-studied drugs ───────────────────

def pristine_analysis(data: dict, df: pd.DataFrame) -> dict:
    """
    Analyze model performance on trials WITH real Open Targets data vs all trials.

    Professor feedback #3: The AUC drop when filtering to OT-characterized drugs
    may indicate that some model performance comes from distinguishing
    well-characterized drugs (which have many features filled) rather than biology alone.

    v6: We explicitly measure and report this distinction.
    """
    print("\n" + "=" * 70)
    print("  PRISTINE vs FULL DATASET ANALYSIS")
    print("  (Addressing: does performance come from biology or data richness?)")
    print("=" * 70)

    label_name = "label_balanced"
    has_ot = df.get("ot_data_missing", pd.Series([1]*len(df), index=df.index)) == 0

    full_train = data["train_df"]
    full_test = data["test_df"]

    pristine_train = full_train[full_train.index.isin(df[has_ot].index)]
    pristine_test = full_test[full_test.index.isin(df[has_ot].index)]

    print(f"\n  FULL DATASET: {len(full_train)} train, {len(full_test)} test")
    print(f"  PRISTINE (has OT): {len(pristine_train)} train, {len(pristine_test)} test")

    if len(pristine_test) < 10:
        print("  Too few pristine test trials for analysis")
        return {}

    y_pristine_test = data["test_labels"][label_name][data["test_labels"][label_name].index.isin(pristine_test.index)]
    y_full_test = data["test_labels"][label_name]

    X_tr, X_te = data["feature_sets"]["composite"]
    X_pristine_te = X_te[X_te.index.isin(pristine_test.index)]
    y_pristine = y_full_test[y_full_test.index.isin(X_pristine_te.index)]

    models = get_models()
    model = models["LightGBM"]

    # Model trained on all data, evaluated on pristine subset
    tr_m = data["train_labels"][label_name].isin([0, 1])
    te_m = y_full_test.isin([0, 1])
    model.fit(X_tr[tr_m], data["train_labels"][label_name][tr_m])
    y_full_clean = y_full_test[te_m]
    X_te_clean = X_te[te_m]
    full_auc = roc_auc_score(y_full_clean, model.predict_proba(X_te_clean)[:, 1])                if len(np.unique(y_full_clean)) > 1 else 0.5

    # Filter -1 indeterminate labels from pristine set
    pristine_clean = y_pristine.isin([0, 1])
    y_pristine_clean = y_pristine[pristine_clean]
    X_pristine_clean = X_pristine_te[X_pristine_te.index.isin(y_pristine_clean.index)]
    if len(X_pristine_clean) >= 10 and len(np.unique(y_pristine_clean)) > 1:
        pristine_auc = roc_auc_score(y_pristine_clean, model.predict_proba(X_pristine_clean)[:, 1])
        print(f"\n  Full dataset AUC:   {full_auc:.3f}")
        print(f"  Pristine-only AUC:  {pristine_auc:.3f}")

        if pristine_auc > full_auc + 0.05:
            print("  ⚠ Pristine AUC > Full AUC → well-characterized drugs easier to classify")
            print("    → Some model performance may reflect data richness, not pure biology")
        elif pristine_auc < full_auc - 0.05:
            print("  ✓ Pristine AUC < Full AUC → biology-only prediction is harder for well-studied targets")
            print("    → Model may be using familiarity/mapping as a proxy signal")
        else:
            print("  ✓ Pristine AUC ≈ Full AUC → consistent signal across characterized/uncharacterized")
    else:
        print(f"  Insufficient pristine test data (n={len(X_pristine_te)})")
        pristine_auc = 0.0

    return {"full_auc": full_auc, "pristine_auc": pristine_auc,
            "n_pristine_test": len(X_pristine_te)}