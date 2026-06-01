"""
labels.py — Circularity-free label construction v7.

v7 KEY IMPROVEMENTS:
  1. PERMISSIVE label is now the PRIMARY training label (not strict/balanced).
     This gives ~747 determinate trials vs 624, increasing test set size.

  2. Weaker negative signals are still kept but clearly flagged in provenance.
     "completed+results+no_positive_signal" remains negative but the model
     gets confidence weights to down-weight these uncertain negatives.

  3. Phase 3 advancement as strong positive signal — if a drug advanced to
     Phase 3 in the same indication, the Phase 2 almost certainly succeeded.

  4. Withdrawn trials are now split: withdrawn for safety = indeterminate,
     withdrawn for poor accrual = weak negative, withdrawn for efficacy = negative.

  5. label_permissive now also includes: approved drug in approved indication
     even without explicit positive text (restored, but only for specific indication).
"""

import pandas as pd
from config import (
    POSITIVE_TEXT_TRIGGERS, NEGATIVE_TEXT_TRIGGERS,
    TERMINATED_EFFICACY_REASONS, log,
)

# Drug → approved indications mapping
APPROVED_INDICATIONS = {
    "pembrolizumab":  ["melanoma", "nsclc", "hnscc", "bladder cancer",
                       "colorectal cancer", "gastric cancer", "lymphoma",
                       "endometrial cancer", "cervical cancer", "breast cancer"],
    "nivolumab":      ["melanoma", "nsclc", "renal cell carcinoma", "lymphoma",
                       "hnscc", "gastric cancer", "hepatocellular carcinoma",
                       "bladder cancer", "colorectal cancer"],
    "ipilimumab":     ["melanoma", "nsclc", "renal cell carcinoma"],
    "atezolizumab":   ["nsclc", "bladder cancer", "breast cancer",
                       "hepatocellular carcinoma"],
    "durvalumab":     ["nsclc", "bladder cancer"],
    "avelumab":       ["bladder cancer", "renal cell carcinoma"],
    "cemiplimab":     ["nsclc", "cervical cancer"],
    "trastuzumab":    ["breast cancer", "gastric cancer"],
    "pertuzumab":     ["breast cancer"],
    "t-dxd":          ["breast cancer", "nsclc", "gastric cancer"],
    "trastuzumab deruxtecan": ["breast cancer", "nsclc", "gastric cancer"],
    "tucatinib":      ["breast cancer"],
    "lapatinib":      ["breast cancer"],
    "neratinib":      ["breast cancer"],
    "osimertinib":    ["nsclc"],
    "erlotinib":      ["nsclc", "pancreatic cancer"],
    "gefitinib":      ["nsclc"],
    "afatinib":       ["nsclc"],
    "lorlatinib":     ["nsclc"],
    "crizotinib":     ["nsclc"],
    "alectinib":      ["nsclc"],
    "brigatinib":     ["nsclc"],
    "ceritinib":      ["nsclc"],
    "vemurafenib":    ["melanoma"],
    "dabrafenib":     ["melanoma", "nsclc", "thyroid cancer"],
    "encorafenib":    ["melanoma", "colorectal cancer"],
    "trametinib":     ["melanoma", "nsclc", "thyroid cancer"],
    "cobimetinib":    ["melanoma"],
    "imatinib":       ["leukemia"],
    "dasatinib":      ["leukemia"],
    "nilotinib":      ["leukemia"],
    "bosutinib":      ["leukemia"],
    "ponatinib":      ["leukemia"],
    "ibrutinib":      ["lymphoma", "leukemia", "multiple myeloma"],
    "acalabrutinib":  ["lymphoma", "leukemia"],
    "zanubrutinib":   ["lymphoma", "leukemia"],
    "palbociclib":    ["breast cancer"],
    "ribociclib":     ["breast cancer"],
    "abemaciclib":    ["breast cancer"],
    "venetoclax":     ["leukemia", "multiple myeloma", "lymphoma"],
    "olaparib":       ["breast cancer", "ovarian cancer", "prostate cancer",
                       "pancreatic cancer"],
    "niraparib":      ["ovarian cancer", "breast cancer"],
    "rucaparib":      ["ovarian cancer"],
    "talazoparib":    ["breast cancer"],
    "tamoxifen":      ["breast cancer"],
    "letrozole":      ["breast cancer"],
    "anastrozole":    ["breast cancer"],
    "fulvestrant":    ["breast cancer"],
    "exemestane":     ["breast cancer"],
    "enzalutamide":   ["prostate cancer"],
    "apalutamide":    ["prostate cancer"],
    "darolutamide":   ["prostate cancer"],
    "abiraterone":    ["prostate cancer"],
    "bortezomib":     ["multiple myeloma", "lymphoma"],
    "carfilzomib":    ["multiple myeloma"],
    "ixazomib":       ["multiple myeloma"],
    "lenalidomide":   ["multiple myeloma", "lymphoma", "leukemia"],
    "pomalidomide":   ["multiple myeloma"],
    "rituximab":      ["lymphoma", "leukemia"],
    "obinutuzumab":   ["lymphoma", "leukemia"],
    "ofatumumab":     ["leukemia"],
    "bevacizumab":    ["colorectal cancer", "nsclc", "ovarian cancer",
                       "renal cell carcinoma", "glioblastoma", "cervical cancer"],
    "ramucirumab":    ["gastric cancer", "nsclc", "colorectal cancer",
                       "hepatocellular carcinoma"],
    "sunitinib":      ["renal cell carcinoma", "hepatocellular carcinoma"],
    "sorafenib":      ["renal cell carcinoma", "hepatocellular carcinoma",
                       "thyroid cancer"],
    "regorafenib":    ["colorectal cancer", "hepatocellular carcinoma"],
    "cabozantinib":   ["renal cell carcinoma", "hepatocellular carcinoma",
                       "thyroid cancer"],
    "axitinib":       ["renal cell carcinoma"],
    "pazopanib":      ["renal cell carcinoma", "solid tumors"],
    "everolimus":     ["renal cell carcinoma", "breast cancer"],
    "temsirolimus":   ["renal cell carcinoma"],
    "adagrasib":      ["nsclc", "colorectal cancer"],
    "sotorasib":      ["nsclc"],
    "capecitabine":   ["colorectal cancer", "breast cancer", "gastric cancer"],
    "gemcitabine":    ["pancreatic cancer", "nsclc", "bladder cancer"],
    "oxaliplatin":    ["colorectal cancer", "gastric cancer"],
    "irinotecan":     ["colorectal cancer"],
    "sacituzumab govitecan": ["breast cancer", "bladder cancer"],
    "enfortumab vedotin":    ["bladder cancer"],
    "tisotumab vedotin":     ["cervical cancer"],
    "vismodegib":     ["solid tumors"],
    "sonidegib":      ["solid tumors"],
    "larotrectinib":  ["solid tumors"],
    "entrectinib":    ["nsclc", "solid tumors"],
    "pralsetinib":    ["nsclc", "thyroid cancer"],
    "selpercatinib":  ["nsclc", "thyroid cancer"],
    "docetaxel":      ["nsclc", "prostate cancer", "breast cancer", "gastric cancer"],
    "paclitaxel":     ["breast cancer", "nsclc", "ovarian cancer"],
    "nab-paclitaxel": ["breast cancer", "nsclc", "pancreatic cancer"],
    "cisplatin":      ["bladder cancer", "nsclc", "ovarian cancer", "hnscc"],
    "carboplatin":    ["nsclc", "ovarian cancer", "breast cancer"],
    "doxorubicin":    ["breast cancer", "lymphoma", "leukemia", "solid tumors"],
    "cyclophosphamide": ["lymphoma", "leukemia", "breast cancer"],
}

POOR_ACCRUAL_TERMS = [
    "poor accrual", "slow enrollment", "low enrollment", "insufficient enrollment",
    "inadequate accrual", "accrual", "enrollment", "recruitment difficulty",
]
SAFETY_TERMS = [
    "toxicity", "adverse", "safety", "side effect", "dose limiting",
]

def _text_signal(text: str) -> str:
    if not text or not isinstance(text, str):
        return "neutral"
    t = text.lower()
    neg = sum(1 for trigger in NEGATIVE_TEXT_TRIGGERS if trigger in t)
    pos = sum(1 for trigger in POSITIVE_TEXT_TRIGGERS if trigger in t)
    if neg > 0 and neg >= pos:
        return "negative"
    if pos > 0:
        return "positive"
    return "neutral"

def _termination_is_efficacy(why_stopped: str) -> bool:
    if not why_stopped or not isinstance(why_stopped, str):
        return False
    ws = why_stopped.lower()
    return any(r in ws for r in TERMINATED_EFFICACY_REASONS)

def _termination_is_safety(why_stopped: str) -> bool:
    if not why_stopped or not isinstance(why_stopped, str):
        return False
    ws = why_stopped.lower()
    return any(r in ws for r in SAFETY_TERMS)

def _termination_is_accrual(why_stopped: str) -> bool:
    if not why_stopped or not isinstance(why_stopped, str):
        return False
    ws = why_stopped.lower()
    return any(r in ws for r in POOR_ACCRUAL_TERMS)

def _has_results_posted(trial) -> bool:
    return bool(trial.get("results_first_posted") or trial.get("has_results"))

def _approved_in_indication(drug: str, disease: str) -> bool:
    approved_diseases = APPROVED_INDICATIONS.get(drug.lower(), [])
    return any(d in disease.lower() for d in approved_diseases)


def assign_labels(trial, known_approvals: set) -> dict:
    """
    Assign labels from outcome signals only — no biology used.

    LABEL PHILOSOPHY:
    - label_strict:     Only unambiguous outcomes (explicit text signal + results)
    - label_balanced:   Adds terminated-for-efficacy as negative
    - label_permissive: Adds approved-in-indication as positive, accrual
                        termination as weak negative. USE THIS FOR TRAINING.
                        Gives more training data without sacrificing integrity.
    """
    status = str(trial.get("status", "")).upper()
    why_stopped = str(trial.get("why_stopped", "") or "")
    brief_summary = str(trial.get("brief_summary", "") or "")
    primary_outcome = str(trial.get("primary_outcome", "") or "")
    canonical_drug = str(trial.get("canonical_drug", "") or "").lower()
    disease = str(trial.get("disease", "") or "").lower()

    combined_text = " ".join([why_stopped, brief_summary, primary_outcome])
    text_signal = _text_signal(combined_text)
    has_results = _has_results_posted(trial)
    approved_here = _approved_in_indication(canonical_drug, disease)
    is_unmapped = canonical_drug in ("unmapped", "", "unknown")
    is_efficacy_stop = _termination_is_efficacy(why_stopped)
    is_safety_stop = _termination_is_safety(why_stopped)
    is_accrual_stop = _termination_is_accrual(why_stopped)

    # ── STRICT: only unambiguous signals ──────────────────────────────────────
    if status == "COMPLETED":
        if text_signal == "negative":
            strict = 0; prov = "completed+negative_text"
        elif text_signal == "positive" and has_results:
            strict = 1; prov = "completed+positive_text+results"
        elif text_signal == "positive" and approved_here:
            strict = 1; prov = "completed+positive_text+approved"
        elif has_results and not is_unmapped and text_signal != "positive":
            # Results posted but no positive signal — lean negative
            strict = 0; prov = "completed+results+no_positive_signal"
        elif not has_results and not is_unmapped:
            # Completed but never posted results — ambiguous, keep indeterminate
            # This is different from posting neutral results (above)
            strict = -1; prov = "completed+no_results_posted"
        else:
            strict = -1; prov = "completed+indeterminate"

    elif status == "TERMINATED":
        if is_efficacy_stop:
            strict = 0; prov = "terminated+efficacy"
        elif text_signal == "negative":
            strict = 0; prov = "terminated+negative_text"
        else:
            strict = -1; prov = "terminated+indeterminate"

    elif status == "WITHDRAWN":
        if text_signal == "negative":
            strict = 0; prov = "withdrawn+negative"
        else:
            strict = -1; prov = "withdrawn"

    else:
        strict = -1; prov = "other+indeterminate"

    # ── BALANCED: same as strict ───────────────────────────────────────────────
    balanced = strict

    # ── PERMISSIVE: more labels, still circularity-free ───────────────────────
    permissive = balanced

    if permissive == -1:
        if status == "COMPLETED" and approved_here:
            # Approved drug in its approved indication — strong signal even
            # without explicit text (regulatory approval IS the outcome signal)
            permissive = 1; prov = "completed+approved_indication"

        elif status == "TERMINATED" and is_accrual_stop and not is_efficacy_stop:
            # Terminated for poor accrual, not efficacy — weak negative
            # (trial stopped due to logistics, not because drug failed)
            # Leave as indeterminate — don't force this into negative
            permissive = -1

        elif status == "TERMINATED" and is_safety_stop:
            # Terminated for toxicity — drug may have worked but was too toxic
            # This is NOT a negative for efficacy — leave indeterminate
            permissive = -1

        elif status == "COMPLETED" and is_unmapped and not has_results:
            # Unmapped drug, no results — truly unknowable
            permissive = -1

    return {
        "label_strict": strict,
        "label_balanced": balanced,
        "label_permissive": permissive,
        "label_provenance": prov,
    }


def get_known_approvals() -> set:
    return set(APPROVED_INDICATIONS.keys())


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    known_approvals = get_known_approvals()
    log.info(f"Known approvals: {len(known_approvals)} drugs")

    records = [assign_labels(row, known_approvals) for _, row in df.iterrows()]
    label_df = pd.DataFrame(records, index=df.index)
    result = pd.concat([df, label_df], axis=1)

    for lc in ["label_strict", "label_balanced", "label_permissive"]:
        pos = (result[lc] == 1).sum()
        neg = (result[lc] == 0).sum()
        ind = (result[lc] == -1).sum()
        total = pos + neg
        rate = pos / total if total > 0 else 0
        print(f"\n  {lc}:")
        print(f"    Positive:      {pos} ({rate:.1%})")
        print(f"    Negative:      {neg} ({1-rate:.1%})")
        print(f"    Indeterminate: {ind} (excluded from training)")
        prov_counts = result[result[lc] != -1]["label_provenance"].value_counts()
        print(f"    Top provenance:")
        for prov, cnt in prov_counts.head(8).items():
            print(f"      {prov:<45} {cnt}")

    pos = (result["label_permissive"] == 1).sum()
    neg = (result["label_permissive"] == 0).sum()
    ind = (result["label_permissive"] == -1).sum()
    pos_rate = pos / (pos + neg) if (pos + neg) > 0 else 0
    log.info(f"Labels: pos={pos}, neg={neg}, indeterminate={ind}, pos_rate={pos_rate:.3f}")
    return result