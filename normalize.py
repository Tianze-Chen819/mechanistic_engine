"""
normalize.py - Entity normalization for trials.

Takes raw trial DataFrame and adds normalized columns:
  - canonical_drug, target_genes, primary_target, modality, moa
  - disease (canonical name)
  - biomarker (extracted from text)
  - confidence scores
"""

import pandas as pd
from drug_target_db import DRUG_TARGET_DB, BIOMARKER_PATTERNS
from config import log


# ============================================================
# DRUG NORMALIZATION
# ============================================================
def normalize_drug(intervention_names, use_chembl_fallback=True):
    """
    Match intervention names to our drug-target database.
    Returns: (canonical_drug, targets_list, modality, moa, confidence)

    Tier 3 (v9 ADDED): if no curated match, try ChEMBL's own mechanism
    database (auto_target_mapping.py) before giving up. This is a documented,
    auditable public-API lookup, not a guess — see that module's docstring
    for why hand-typing more curated entries wasn't the right fix for the
    corpus's "mostly already-validated drugs" skew. Confidence is capped at
    0.6 (below curated matches) and the moa is tagged "chembl_auto" so
    downstream analysis can filter these out if a stricter subset is wanted.
    Set use_chembl_fallback=False to reproduce the old (curated-dict-only)
    behaviour, e.g. for fast iteration without network calls.
    """
    for name in intervention_names:
        name_lower = name.lower().strip()

        # Direct match
        if name_lower in DRUG_TARGET_DB:
            info = DRUG_TARGET_DB[name_lower]
            return name_lower, info["targets"], info["modality"], info["moa"], 0.95

        # Substring match (drug name found inside intervention name, or vice versa)
        for drug_key, info in DRUG_TARGET_DB.items():
            if drug_key in name_lower or name_lower in drug_key:
                return drug_key, info["targets"], info["modality"], info["moa"], 0.85

    if use_chembl_fallback:
        from auto_target_mapping import chembl_lookup
        for name in intervention_names:
            entry = chembl_lookup(name)
            if entry:
                return name.lower().strip(), entry["targets"], entry["modality"], \
                    entry["moa"], entry["confidence"]

    return "unmapped", [], "unknown", "unknown", 0.10


# ============================================================
# BIOMARKER EXTRACTION
# ============================================================
def extract_biomarker(title, eligibility):
    """Extract biomarker mentions from trial title + eligibility text."""
    text = (str(title) + " " + str(eligibility)).lower()
    found = []
    for bmk, patterns in BIOMARKER_PATTERNS.items():
        if any(p in text for p in patterns):
            found.append(bmk)
    return found[0] if found else None


# ============================================================
# DISEASE NORMALIZATION
# ============================================================
# Priority-ordered: more specific patterns first
DISEASE_MAP = [
    ("non-small cell lung", "NSCLC", 0.95),
    ("non small cell lung", "NSCLC", 0.95),
    ("nsclc", "NSCLC", 0.95),
    ("small cell lung", "SCLC", 0.90),
    ("triple-negative breast", "TNBC", 0.95),
    ("triple negative breast", "TNBC", 0.95),
    ("her2-positive breast", "HER2+ breast cancer", 0.95),
    ("her2+ breast", "HER2+ breast cancer", 0.95),
    ("breast cancer", "breast cancer", 0.85),
    ("breast neoplasm", "breast cancer", 0.85),
    ("colorectal", "colorectal cancer", 0.90),
    ("colon cancer", "colorectal cancer", 0.90),
    ("rectal cancer", "colorectal cancer", 0.90),
    ("pancreatic", "pancreatic cancer", 0.90),
    ("gastric", "gastric cancer", 0.90),
    ("stomach cancer", "gastric cancer", 0.90),
    ("melanoma", "melanoma", 0.95),
    ("renal cell", "renal cell carcinoma", 0.90),
    ("kidney cancer", "renal cell carcinoma", 0.85),
    ("hepatocellular", "hepatocellular carcinoma", 0.90),
    ("liver cancer", "hepatocellular carcinoma", 0.85),
    ("ovarian", "ovarian cancer", 0.90),
    ("prostate", "prostate cancer", 0.90),
    ("bladder", "bladder cancer", 0.90),
    ("urothelial", "bladder cancer", 0.90),
    ("head and neck", "HNSCC", 0.90),
    ("glioblastoma", "glioblastoma", 0.95),
    ("glioma", "glioma", 0.85),
    ("acute myeloid leukemia", "AML", 0.95),
    ("aml", "AML", 0.90),
    ("chronic lymphocytic leukemia", "CLL", 0.95),
    ("cll", "CLL", 0.90),
    ("diffuse large b-cell", "DLBCL", 0.95),
    ("dlbcl", "DLBCL", 0.90),
    ("lymphoma", "lymphoma", 0.80),
    ("multiple myeloma", "multiple myeloma", 0.95),
    ("myeloma", "multiple myeloma", 0.90),
    ("cholangiocarcinoma", "cholangiocarcinoma", 0.90),
    ("endometrial", "endometrial cancer", 0.90),
    ("thyroid", "thyroid cancer", 0.85),
    ("sarcoma", "sarcoma", 0.85),
    ("mesothelioma", "mesothelioma", 0.90),
    ("lung", "lung cancer NOS", 0.75),
    ("solid tumor", "solid tumors", 0.60),
    ("advanced cancer", "solid tumors", 0.50),
    ("neoplasm", "cancer NOS", 0.40),
    ("cancer", "cancer NOS", 0.40),
    ("carcinoma", "cancer NOS", 0.40),
]


def normalize_disease(conditions):
    """Normalize disease/condition names to canonical oncology terms."""
    text = " ".join(str(c).lower() for c in conditions) if isinstance(conditions, list) else str(conditions).lower()

    for pattern, canonical, conf in DISEASE_MAP:
        if pattern in text:
            return canonical, conf

    return "cancer NOS", 0.20


# ============================================================
# MAIN NORMALIZATION FUNCTION
# ============================================================
def normalize_all(trials_df):
    """
    Apply drug, disease, and biomarker normalization to all trials.
    Adds columns in-place and returns the DataFrame.
    """
    print("  Normalizing drugs...")
    norm_results = trials_df["intervention_names"].apply(normalize_drug)
    trials_df["canonical_drug"] = norm_results.apply(lambda x: x[0])
    trials_df["target_genes"] = norm_results.apply(lambda x: x[1])
    trials_df["primary_target"] = norm_results.apply(lambda x: x[1][0] if x[1] else "UNKNOWN")
    trials_df["modality"] = norm_results.apply(lambda x: x[2])
    trials_df["moa"] = norm_results.apply(lambda x: x[3])
    trials_df["drug_confidence"] = norm_results.apply(lambda x: x[4])

    print("  Normalizing diseases...")
    disease_results = trials_df["condition"].apply(normalize_disease)
    trials_df["disease"] = disease_results.apply(lambda x: x[0])
    trials_df["disease_confidence"] = disease_results.apply(lambda x: x[1])

    print("  Extracting biomarkers...")
    trials_df["biomarker"] = trials_df.apply(
        lambda r: extract_biomarker(r["title"], r["eligibility_text"]), axis=1
    )

    # Summary
    mapped = trials_df[trials_df["canonical_drug"] != "unmapped"]
    print(f"\n  Normalization complete:")
    print(f"    Total trials: {len(trials_df)}")
    print(f"    Drug mapped: {len(mapped)} ({len(mapped)/len(trials_df)*100:.1f}%)")
    print(f"    Unique drugs: {mapped['canonical_drug'].nunique()}")
    print(f"    Unique targets: {mapped['primary_target'].nunique()}")
    print(f"    Unique diseases: {trials_df['disease'].nunique()}")
    print(f"    Trials with biomarker: {trials_df['biomarker'].notna().sum()}")

    log.info(f"Normalization: {len(mapped)}/{len(trials_df)} drugs mapped, "
             f"{trials_df['biomarker'].notna().sum()} with biomarkers")

    return trials_df


# ============================================================
# ENTRY POINT (for testing this file alone)
# ============================================================
if __name__ == "__main__":
    print("Testing normalize.py...")
    # Quick test with fake data
    test_df = pd.DataFrame({
        "intervention_names": [["pembrolizumab"], ["some_random_drug"]],
        "condition": [["Non-Small Cell Lung Cancer"], ["Advanced Cancer"]],
        "title": ["PD-L1 positive NSCLC trial", "Phase 2 cancer study"],
        "eligibility_text": ["EGFR mutation required", "open enrollment"],
    })
    result = normalize_all(test_df)
    print(result[["canonical_drug", "primary_target", "disease", "biomarker"]].to_string())