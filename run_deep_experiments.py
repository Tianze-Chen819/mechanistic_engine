"""
Standalone runner for the optional deep-learning branch.

Run this after the baseline pipeline has created the feature matrix:
    python run_pipeline.py
"""

from __future__ import annotations

import argparse
import sys

from config import ALL_RAW_FEATURES, COMPOSITE_SCORES, DATA_DIR, DEEP_FEATURE_SET, DEEP_LABEL_COL


MISSING_DATA_MESSAGE = (
    "Required modeling outputs were not found. "
    "Run `python run_pipeline.py` first, then rerun this script."
)


def _load_feature_matrix():
    csv_path = DATA_DIR / "full_feature_matrix.csv"
    parquet_path = DATA_DIR / "full_feature_matrix.parquet"

    if not parquet_path.exists() and not csv_path.exists():
        print(MISSING_DATA_MESSAGE)
        raise SystemExit(1)

    import pandas as pd

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(csv_path)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c in ALL_RAW_FEATURES
        or c in COMPOSITE_SCORES
        or c.endswith("_missing")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run optional deep-learning experiments.")
    parser.add_argument("--smoke", action="store_true", help="Run 1-2 epoch lightweight smoke test.")
    parser.add_argument(
        "--train-tree-baseline",
        action="store_true",
        help="Retrain tree baselines in standalone mode for ensemble comparison.",
    )
    args = parser.parse_args()

    try:
        from modeling_deep import run_deep_experiments
    except ImportError as exc:
        print(f"PyTorch deep-learning dependencies are missing: {exc}")
        print("Install them with `.venv/bin/pip install -r requirements-deep.txt`.")
        return 1

    df = _load_feature_matrix()
    from embedding_features import apply_precomputed_cbio_features

    df = apply_precomputed_cbio_features(df)
    required = {"label_permissive", "start_year", "canonical_drug", "primary_target", "disease", "modality"}
    missing = sorted(required.difference(df.columns))
    if missing:
        print(MISSING_DATA_MESSAGE)
        print(f"Missing required columns: {missing}")
        return 1

    feature_cols = _feature_columns(df)
    if not feature_cols:
        print(MISSING_DATA_MESSAGE)
        print("No recognized biology feature columns were found.")
        return 1

    from run_pipeline import build_modeling_data

    data = build_modeling_data(df, feature_cols)
    tree_model_output = None
    if args.train_tree_baseline and not args.smoke:
        from modeling import train_all_models

        tree_model_output = train_all_models(data)

    run_deep_experiments(
        df_features=df,
        modeling_data=data,
        tree_model_output=tree_model_output,
        feature_set=DEEP_FEATURE_SET,
        label_col=DEEP_LABEL_COL,
        smoke=args.smoke,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
