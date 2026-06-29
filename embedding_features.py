"""
Utilities for entity vocabularies and numeric biology features.

The first deep-learning branch uses lightweight learned categorical embeddings.
External pretrained embeddings can be plugged in later through
ExternalEmbeddingProvider without making ChemBERTa, ESM, RDKit, or transformers
default dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR
from drug_target_db import DRUG_TARGET_DB


ENTITY_COLUMNS = ["canonical_drug", "primary_target", "disease", "modality", "moa"]
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def _clean_token(value) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def infer_moa(canonical_drug: str) -> str:
    """Map canonical drug name to mechanism tag from the curated target DB."""
    key = _clean_token(canonical_drug).lower()
    if key in ("unknown", "unmapped"):
        return "unknown"
    info = DRUG_TARGET_DB.get(key)
    if not info:
        return "unknown"
    return _clean_token(info.get("moa", "unknown"))


def ensure_moa_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a moa column, deriving it from canonical_drug if needed."""
    result = df.copy()
    if "moa" not in result.columns:
        drugs = result["canonical_drug"] if "canonical_drug" in result.columns else pd.Series(["unknown"] * len(result), index=result.index)
        result["moa"] = drugs.apply(infer_moa)
    else:
        missing = result["moa"].isna() | (result["moa"].astype(str).str.strip() == "")
        if missing.any():
            result.loc[missing, "moa"] = result.loc[missing, "canonical_drug"].apply(infer_moa)
    return result


@dataclass
class EntityVocab:
    """Vocabulary for one entity column with stable reserved IDs."""

    name: str
    token_to_id: dict[str, int] = field(default_factory=lambda: {PAD_TOKEN: 0, UNK_TOKEN: 1})

    def fit(self, values: Iterable) -> "EntityVocab":
        tokens = sorted({_clean_token(v) for v in values})
        for token in tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)
        return self

    def transform(self, values: Iterable) -> np.ndarray:
        unk_id = self.token_to_id[UNK_TOKEN]
        ids = [self.token_to_id.get(_clean_token(v), unk_id) for v in values]
        return np.asarray(ids, dtype=np.int64)

    @property
    def size(self) -> int:
        return len(self.token_to_id)


@dataclass
class EntityVocabBundle:
    """Vocabularies for trial entity tokens."""

    columns: list[str] = field(default_factory=lambda: ENTITY_COLUMNS.copy())
    vocabs: dict[str, EntityVocab] = field(default_factory=dict)

    def fit(self, train_df: pd.DataFrame) -> "EntityVocabBundle":
        df = ensure_moa_column(train_df)
        self.vocabs = {}
        for col in self.columns:
            vocab = EntityVocab(col)
            vocab.fit(df[col] if col in df.columns else ["unknown"] * len(df))
            self.vocabs[col] = vocab
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        prepared = ensure_moa_column(df)
        arrays = []
        for col in self.columns:
            values = prepared[col] if col in prepared.columns else ["unknown"] * len(prepared)
            arrays.append(self.vocabs[col].transform(values))
        return np.stack(arrays, axis=1)

    @property
    def vocab_sizes(self) -> dict[str, int]:
        return {col: self.vocabs[col].size for col in self.columns}


@dataclass
class BiologyFeatureScaler:
    """Train-only scaler for biology feature matrices."""

    columns: list[str] | None = None
    scaler: StandardScaler = field(default_factory=StandardScaler)

    def fit(self, train_matrix: pd.DataFrame) -> "BiologyFeatureScaler":
        self.columns = list(train_matrix.columns)
        clean = train_matrix.replace([np.inf, -np.inf], np.nan).fillna(-1.0)
        self.scaler.fit(clean)
        return self

    def transform(self, matrix: pd.DataFrame) -> np.ndarray:
        if self.columns is None:
            raise ValueError("BiologyFeatureScaler must be fit before transform.")
        aligned = matrix.reindex(columns=self.columns, fill_value=-1.0)
        clean = aligned.replace([np.inf, -np.inf], np.nan).fillna(-1.0)
        return self.scaler.transform(clean).astype(np.float32)


class ExternalEmbeddingProvider:
    """
    Future extension point for pretrained vectors.

    The default implementation intentionally returns None so the local baseline
    has no large model dependencies. Later implementations can load local
    ChemBERTa, ESM, disease ontology, or graph embedding files behind this API.
    """

    enabled = False

    def get_drug_vector(self, canonical_drug: str):
        return None

    def get_target_vector(self, primary_target: str):
        return None

    def get_disease_vector(self, disease: str):
        return None


class LocalExternalEmbeddingProvider(ExternalEmbeddingProvider):
    """Load optional local pretrained vectors from CSV files."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.drug_vectors = self._load_vectors("external_drug_embeddings.csv")
        self.target_vectors = self._load_vectors("external_target_embeddings.csv")
        self.disease_vectors = self._load_vectors("external_disease_embeddings.csv")
        self.enabled = any([self.drug_vectors, self.target_vectors, self.disease_vectors])

    def _load_vectors(self, filename: str) -> dict[str, np.ndarray]:
        path = self.data_dir / filename
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        if df.shape[1] < 2:
            return {}
        entity_col = df.columns[0]
        value_cols = list(df.columns[1:])
        vectors = {}
        for _, row in df.iterrows():
            key = _clean_token(row[entity_col]).lower()
            values = pd.to_numeric(row[value_cols], errors="coerce").fillna(0.0)
            vectors[key] = values.to_numpy(dtype=np.float32)
        return vectors

    def _lookup(self, vectors: dict[str, np.ndarray], key: str):
        if not vectors:
            return None
        sample_dim = len(next(iter(vectors.values())))
        return vectors.get(_clean_token(key).lower(), np.zeros(sample_dim, dtype=np.float32))

    def get_drug_vector(self, canonical_drug: str):
        return self._lookup(self.drug_vectors, canonical_drug)

    def get_target_vector(self, primary_target: str):
        return self._lookup(self.target_vectors, primary_target)

    def get_disease_vector(self, disease: str):
        return self._lookup(self.disease_vectors, disease)


def build_external_embedding_matrix(df: pd.DataFrame, provider: ExternalEmbeddingProvider) -> np.ndarray:
    """Build concatenated local external vectors; returns 0 columns when disabled."""
    if not getattr(provider, "enabled", False):
        return np.zeros((len(df), 0), dtype=np.float32)

    rows = []
    for _, row in df.iterrows():
        pieces = [
            provider.get_drug_vector(row.get("canonical_drug", "unknown")),
            provider.get_target_vector(row.get("primary_target", "unknown")),
            provider.get_disease_vector(row.get("disease", "unknown")),
        ]
        rows.append(np.concatenate([p for p in pieces if p is not None]).astype(np.float32))
    return np.vstack(rows) if rows else np.zeros((0, 0), dtype=np.float32)


def apply_precomputed_cbio_features(
    df: pd.DataFrame,
    path: Path = DATA_DIR / "cbioportal_mutation_summary.csv",
) -> pd.DataFrame:
    """Merge optional local cBioPortal mutation summaries into feature rows."""
    if not path.exists():
        return df

    required = {"disease", "primary_target", "alteration_frequency", "sample_count", "study_source"}
    summary = pd.read_csv(path)
    if not required.issubset(summary.columns):
        missing = sorted(required.difference(summary.columns))
        raise ValueError(f"cBioPortal summary missing columns: {missing}")

    result = df.copy()
    summary = summary.copy()
    summary["_disease_key"] = summary["disease"].map(lambda v: _clean_token(v).lower())
    summary["_target_key"] = summary["primary_target"].map(lambda v: _clean_token(v).upper())
    result["_disease_key"] = result["disease"].map(lambda v: _clean_token(v).lower())
    result["_target_key"] = result["primary_target"].map(lambda v: _clean_token(v).upper())

    merge_cols = ["_disease_key", "_target_key", "alteration_frequency", "sample_count", "study_source"]
    merged = result.merge(
        summary[merge_cols],
        on=["_disease_key", "_target_key"],
        how="left",
        suffixes=("", "_cbio_precomputed"),
    )
    has_local = merged["alteration_frequency_cbio_precomputed"].notna()
    if has_local.any():
        merged.loc[has_local, "alteration_frequency"] = merged.loc[has_local, "alteration_frequency_cbio_precomputed"]
        if "cbio_data_missing" in merged.columns:
            merged.loc[has_local, "cbio_data_missing"] = 0
        if "lineage_specificity" in merged.columns:
            lineage_missing = merged["lineage_specificity"].isna() | (merged["lineage_specificity"] <= 0)
            fill_mask = has_local & lineage_missing
            merged.loc[fill_mask, "lineage_specificity"] = merged.loc[fill_mask, "alteration_frequency_cbio_precomputed"]
        merged["cbio_precomputed_sample_count"] = merged["sample_count"]
        merged["cbio_precomputed_study_source"] = merged["study_source"]

    drop_cols = [
        "_disease_key", "_target_key", "alteration_frequency_cbio_precomputed",
        "sample_count", "study_source",
    ]
    return merged.drop(columns=[c for c in drop_cols if c in merged.columns])


def build_unmapped_drug_report(df: pd.DataFrame, test_index: pd.Index, label_col: str) -> pd.DataFrame:
    """Summarize unmapped interventions for manual curation prioritization."""
    work = df.copy()
    work["appears_in_test_set"] = work.index.isin(test_index)
    target_unknown = work.get("primary_target", "UNKNOWN").astype(str).str.upper().eq("UNKNOWN")
    unmapped = work[(work.get("canonical_drug", "unmapped") == "unmapped") | target_unknown].copy()
    if unmapped.empty:
        return pd.DataFrame(columns=[
            "intervention_names", "disease", "modality", "label", "target_is_unknown",
            "appears_in_test_set", "frequency_count",
        ])

    unmapped["target_is_unknown"] = unmapped.get("primary_target", "UNKNOWN").astype(str).str.upper().eq("UNKNOWN")
    group_cols = ["intervention_names", "disease", "modality", label_col, "target_is_unknown", "appears_in_test_set"]
    available = [c for c in group_cols if c in unmapped.columns]
    report = (
        unmapped.groupby(available, dropna=False)
        .size()
        .reset_index(name="frequency_count")
        .rename(columns={label_col: "label"})
        .sort_values(["appears_in_test_set", "frequency_count"], ascending=[False, False])
    )
    return report
