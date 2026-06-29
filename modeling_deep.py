"""
Optional deep-learning experiments for trial success prediction.

This module reuses the existing temporal split and biology features. It does not
construct labels and does not replace the tree-based baselines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    DEEP_BATCH_SIZE,
    DEEP_EPOCHS,
    DEEP_LR,
    DEEP_PATIENCE,
    DEEP_WEIGHT_DECAY,
    RANDOM_SEED,
    REPORT_DIR,
    USE_PRETRAINED_EMBEDDINGS,
    log,
)
from deep_models import (
    EmbeddingMLP,
    TrialTensorDataset,
    TrialTransformer,
    classification_metrics,
    predict_probs,
    set_torch_seed,
    train_model,
)
from embedding_features import BiologyFeatureScaler, EntityVocabBundle, ensure_moa_column
from embedding_features import (
    LocalExternalEmbeddingProvider,
    apply_precomputed_cbio_features,
    build_external_embedding_matrix,
    build_unmapped_drug_report,
)


def chronological_validation_split(train_df: pd.DataFrame, label_col: str) -> tuple[pd.Index, pd.Index]:
    """Split training rows into older train and latest-year validation rows."""
    clean = train_df[train_df[label_col].isin([0, 1])].copy()
    clean = clean.sort_values(["start_year", "nct_id"], na_position="first")
    years = [int(y) for y in sorted(clean["start_year"].dropna().unique())]

    for year in reversed(years):
        val_idx = clean.index[clean["start_year"] >= year]
        train_idx = clean.index.difference(val_idx)
        if _has_two_classes(clean.loc[train_idx, label_col]) and _has_two_classes(clean.loc[val_idx, label_col]):
            return train_idx, val_idx

    cutoff = max(1, int(len(clean) * 0.8))
    train_idx = clean.index[:cutoff]
    val_idx = clean.index[cutoff:]
    if not _has_two_classes(clean.loc[val_idx, label_col]):
        raise ValueError("Chronological validation split has fewer than two classes.")
    return train_idx, val_idx


def _has_two_classes(y: pd.Series) -> bool:
    return len(pd.Series(y).dropna().unique()) >= 2


def _limit_index_preserving_classes(index: pd.Index, labels: pd.Series, max_rows: int) -> pd.Index:
    if len(index) <= max_rows:
        return index
    selected = index[:max_rows]
    if _has_two_classes(labels.loc[selected]):
        return selected

    pieces = []
    per_class = max(1, max_rows // 2)
    for _, class_labels in labels.loc[index].groupby(labels.loc[index]):
        pieces.extend(class_labels.index[:per_class])
    selected = pd.Index(pieces[:max_rows])
    return selected if _has_two_classes(labels.loc[selected]) else index


def _make_dataset(df, X, labels, vocab_bundle, scaler, external_matrix=None) -> TrialTensorDataset:
    entity_ids = vocab_bundle.transform(df)
    biology = scaler.transform(X)
    if external_matrix is not None and external_matrix.shape[1] > 0:
        biology = np.concatenate([biology, external_matrix.astype(np.float32)], axis=1)
    return TrialTensorDataset(entity_ids, biology, labels.astype(int).to_numpy())


def _metric_row(model_name, feature_set, label_col, y_true, probs, n_train, n_val, n_test):
    row = {
        "model": model_name,
        "feature_set": feature_set,
        "label_col": label_col,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
    }
    row.update(classification_metrics(y_true, probs))
    return row


def _prediction_metrics(model, dataset, labels, batch_size):
    probs = predict_probs(model, dataset, batch_size=batch_size)
    metrics = classification_metrics(labels, probs)
    metrics["auc_reversed"] = classification_metrics(labels, 1.0 - probs)["roc_auc"]
    return probs, metrics


def _subgroup_rows(model_name, split_name, df, labels, probs, min_n=20):
    rows = []

    def add_group(group_name, values):
        series = pd.Series(values, index=df.index)
        for value, idx in series.groupby(series, dropna=False).groups.items():
            idx = pd.Index(idx)
            if len(idx) < min_n:
                continue
            y = labels.loc[idx].astype(int).to_numpy()
            if len(np.unique(y)) < 2:
                continue
            p = probs[df.index.get_indexer(idx)]
            row = {
                "model": model_name,
                "split": split_name,
                "subgroup": group_name,
                "value": value,
                "n": len(idx),
            }
            row.update(classification_metrics(y, p))
            rows.append(row)

    target = df.get("primary_target", pd.Series("UNKNOWN", index=df.index)).astype(str).str.upper()
    mapped = target.ne("UNKNOWN")
    add_group("mapped_status", np.where(mapped, "mapped", "unmapped"))

    if "is_io_trial" in df.columns:
        add_group("io_status", np.where(df["is_io_trial"].fillna(False).astype(bool), "io", "non_io"))

    if "modality" in df.columns:
        modality = df["modality"].fillna("unknown").astype(str)
        modality_group = np.where(
            modality.isin(["antibody", "adc", "cell_therapy", "bispecific"]),
            modality,
            np.where(modality.eq("small_molecule"), "small_molecule", "other_or_unknown"),
        )
        add_group("modality_group", modality_group)

    if "disease" in df.columns:
        disease_counts = df["disease"].value_counts(dropna=False)
        major = df["disease"].where(df["disease"].map(disease_counts) >= min_n, "other")
        add_group("major_disease", major)

    return rows


def _tree_probabilities(tree_model_output, modeling_data, label_col):
    if not tree_model_output:
        return None, None
    best = tree_model_output.get("best", {}).get(label_col)
    if not best or not best.get("model"):
        return None, None

    feature_set = best["features"]
    model = best["model"]
    X_test = modeling_data["feature_sets"][feature_set][1]
    y_test = modeling_data["test_labels"][label_col]
    clean_mask = y_test.isin([0, 1])
    probs = model.predict_proba(X_test[clean_mask])[:, 1]
    return probs, {
        "model": best["model_name"],
        "feature_set": feature_set,
        "label_col": label_col,
        "roc_auc": best.get("auc"),
        "brier": best.get("brier"),
    }


def run_deep_experiments(
    df_features: pd.DataFrame,
    modeling_data: dict,
    tree_model_output: dict | None = None,
    feature_set: str = "composite",
    label_col: str = "label_permissive",
    smoke: bool = False,
) -> dict:
    """
    Train and evaluate the learned-embedding MLP and small TrialTransformer.
    """
    if feature_set not in modeling_data["feature_sets"]:
        raise ValueError(f"Unknown feature set {feature_set!r}.")

    set_torch_seed(RANDOM_SEED)
    epochs = min(2, DEEP_EPOCHS) if smoke else DEEP_EPOCHS
    patience = min(2, DEEP_PATIENCE) if smoke else DEEP_PATIENCE

    train_df_all = ensure_moa_column(modeling_data["train_df"])
    test_df_all = ensure_moa_column(modeling_data["test_df"])
    X_train_all, X_test_all = modeling_data["feature_sets"][feature_set]
    y_train_all = modeling_data["train_labels"][label_col]
    y_test_all = modeling_data["test_labels"][label_col]

    test_mask = y_test_all.isin([0, 1])
    if not _has_two_classes(y_test_all[test_mask]):
        raise ValueError("Post-cutoff test set has fewer than two classes.")

    train_idx, val_idx = chronological_validation_split(train_df_all.assign(**{label_col: y_train_all}), label_col)
    if smoke:
        train_idx = _limit_index_preserving_classes(train_idx, y_train_all, 160)
        val_idx = _limit_index_preserving_classes(val_idx, y_train_all, 80)

    vocab_bundle = EntityVocabBundle().fit(train_df_all.loc[train_idx])
    scaler = BiologyFeatureScaler().fit(X_train_all.loc[train_idx])
    provider = LocalExternalEmbeddingProvider() if USE_PRETRAINED_EMBEDDINGS else None
    train_external = build_external_embedding_matrix(train_df_all.loc[train_idx], provider) if provider else None
    val_external = build_external_embedding_matrix(train_df_all.loc[val_idx], provider) if provider else None
    test_external = build_external_embedding_matrix(test_df_all.loc[test_mask], provider) if provider else None

    train_dataset = _make_dataset(
        train_df_all.loc[train_idx],
        X_train_all.loc[train_idx],
        y_train_all.loc[train_idx],
        vocab_bundle,
        scaler,
        train_external,
    )
    val_dataset = _make_dataset(
        train_df_all.loc[val_idx],
        X_train_all.loc[val_idx],
        y_train_all.loc[val_idx],
        vocab_bundle,
        scaler,
        val_external,
    )
    test_dataset = _make_dataset(
        test_df_all.loc[test_mask],
        X_test_all.loc[test_mask],
        y_test_all.loc[test_mask],
        vocab_bundle,
        scaler,
        test_external,
    )

    biology_dim = train_dataset.biology.shape[1]
    y_train = y_train_all.loc[train_idx].astype(int).to_numpy()
    y_val = y_train_all.loc[val_idx].astype(int).to_numpy()
    y_test = y_test_all.loc[test_mask].astype(int).to_numpy()
    metrics = []
    subgroup_rows = []
    predictions = test_df_all.loc[test_mask, [
        c for c in ["nct_id", "start_year", "canonical_drug", "primary_target", "disease"]
        if c in test_df_all.columns
    ]].copy()
    predictions["label"] = y_test

    model_specs = [
        ("BiologyOnlyMLP", EmbeddingMLP(vocab_bundle.vocab_sizes, biology_dim, use_entities=False, use_biology=True)),
        ("EntityOnlyMLP", EmbeddingMLP(vocab_bundle.vocab_sizes, biology_dim, use_entities=True, use_biology=False)),
        ("DeepMLP", EmbeddingMLP(vocab_bundle.vocab_sizes, biology_dim, use_entities=True, use_biology=True)),
        ("TrialTransformer", TrialTransformer(vocab_bundle.vocab_sizes, biology_dim)),
    ]

    trained = {}
    for model_name, model in model_specs:
        result = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_labels=y_train,
            epochs=epochs,
            patience=patience,
            batch_size=DEEP_BATCH_SIZE,
            lr=DEEP_LR,
            weight_decay=DEEP_WEIGHT_DECAY,
            seed=RANDOM_SEED,
        )
        train_probs, train_metrics = _prediction_metrics(result.model, train_dataset, y_train, DEEP_BATCH_SIZE)
        val_probs, val_metrics = _prediction_metrics(result.model, val_dataset, y_val, DEEP_BATCH_SIZE)
        probs, test_metrics = _prediction_metrics(result.model, test_dataset, y_test, DEEP_BATCH_SIZE)
        prob_col = {
            "BiologyOnlyMLP": "biology_mlp_prob",
            "EntityOnlyMLP": "entity_mlp_prob",
            "DeepMLP": "mlp_prob",
            "TrialTransformer": "transformer_prob",
        }[model_name]
        predictions[prob_col] = probs
        row = _metric_row(model_name, f"{feature_set}_plus_embeddings", label_col, y_test, probs, len(train_dataset), len(val_dataset), len(test_dataset))
        row.update({
            "train_auc": train_metrics["roc_auc"],
            "validation_auc": val_metrics["roc_auc"],
            "test_auc_reversed": test_metrics["auc_reversed"],
            "train_auc_reversed": train_metrics["auc_reversed"],
            "validation_auc_reversed": val_metrics["auc_reversed"],
        })
        row["best_epoch"] = result.best_epoch
        row["best_val_auc"] = round(result.best_val_auc, 6)
        metrics.append(row)
        trained[model_name] = {"model": result.model, "test_probs": probs, "metrics": row}
        val_df = train_df_all.loc[val_idx].copy()
        test_df = test_df_all.loc[test_mask].copy()
        subgroup_rows.extend(_subgroup_rows(model_name, "pre_2015_validation", val_df, y_train_all.loc[val_idx], val_probs))
        subgroup_rows.extend(_subgroup_rows(model_name, "post_2015_test", test_df, y_test_all.loc[test_mask], probs))

    for col in ["biology_mlp_prob", "entity_mlp_prob", "mlp_prob", "transformer_prob"]:
        if col not in predictions.columns:
            predictions[col] = np.nan

    deep_results = pd.DataFrame(metrics).sort_values("roc_auc", ascending=False)
    deep_results.to_csv(REPORT_DIR / "deep_model_comparison.csv", index=False)
    predictions.to_csv(REPORT_DIR / "deep_predictions.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(REPORT_DIR / "deep_subgroup_metrics.csv", index=False)
    full_df = pd.concat([train_df_all, test_df_all], axis=0)
    build_unmapped_drug_report(full_df, test_df_all.index, label_col).to_csv(
        REPORT_DIR / "unmapped_drug_report.csv", index=False
    )

    ensemble_rows = []
    if tree_model_output and "results" in tree_model_output:
        tree_results = tree_model_output["results"]
        if isinstance(tree_results, pd.DataFrame) and "label" in tree_results.columns:
            subset = tree_results[tree_results["label"] == label_col].copy()
            subset = subset.rename(columns={"label": "label_col", "features": "feature_set"})
            subset["source"] = "tree_baseline"
            ensemble_rows.extend(subset.to_dict("records"))

    for row in metrics:
        deep_row = row.copy()
        deep_row["source"] = "deep"
        ensemble_rows.append(deep_row)

    tree_probs, tree_info = _tree_probabilities(tree_model_output, modeling_data, label_col)
    if tree_probs is not None and len(tree_probs) == len(y_test):
        for deep_name in ["DeepMLP", "TrialTransformer"]:
            deep_probs = trained[deep_name]["test_probs"]
            ensemble_probs = 0.5 * tree_probs + 0.5 * deep_probs
            ensemble_row = _metric_row(
                f"FixedAverage({tree_info['model']}+{deep_name})",
                f"{tree_info['feature_set']}+{feature_set}_plus_embeddings",
                label_col,
                y_test,
                ensemble_probs,
                len(train_dataset),
                len(val_dataset),
                len(test_dataset),
            )
            ensemble_row["source"] = "fixed_average_ensemble"
            ensemble_rows.append(ensemble_row)
    else:
        print("  Tree+deep ensemble requires integrated RUN_DEEP_EXPERIMENTS=1 mode with a fitted tree model.")

    ensemble_df = pd.DataFrame(ensemble_rows)
    ensemble_df.to_csv(REPORT_DIR / "ensemble_comparison.csv", index=False)

    log.info(f"Deep experiments complete: {len(deep_results)} models evaluated")
    print("\n  === DEEP MODEL RESULTS ===")
    print(deep_results[["model", "feature_set", "roc_auc", "pr_auc", "f1", "brier", "balanced_accuracy"]].to_string(index=False))
    return {
        "deep_results": deep_results,
        "predictions": predictions,
        "ensemble_results": ensemble_df,
        "trained": trained,
    }
