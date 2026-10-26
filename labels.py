"""
labels.py — Circularity-free label construction v9.

v9 REWRITE (see docs/label_audit.md for the audit that drove this).

The v7 labeller scored a trial "positive" whenever a phrase like "complete
response" or "approved" appeared anywhere in why_stopped + brief_summary +
primary_outcome. brief_summary and primary_outcome are written when the trial
is REGISTERED, before a single patient enrols — primary_outcome is usually
just the endpoint's textbook definition ("Complete Response (CR): disappearance
of all target lesions..."), and brief_summary background prose routinely
mentions OTHER drugs being "approved". An audit of all 123 v7 positives found
100% of them fired only on those two pre-trial fields — never on why_stopped —
so the label was measuring how a protocol was worded, not what happened.

v9 CHANGES:
  1. Free-text triggers now only look at `why_stopped`, the one field that is
     ever written after the trial concludes. brief_summary and primary_outcome
     are no longer used as label evidence at all.
  2. The `completed+results+no_positive_signal` bucket (93% of the old negative
     class) is no longer forced to negative. A trial merely posting results
     with no post-hoc text to read is genuinely unknown from this dataset
     alone — it is now indeterminate. See PRIORITY 1/2 in labels_v8.py for how
     to actually resolve these (structured p-values, linked publications).
  3. TERMINATED_EFFICACY_REASONS no longer contains the bare word "efficacy"
     (config.py) — it was matching "stopped early due to overwhelming
     efficacy" (a SUCCESS) the same as "terminated for lack of efficacy" (a
     FAILURE). TERMINATED_SUCCESS_REASONS (new, config.py) now identifies the
     success case explicitly.
  4. `known_approvals` is now actually used (previously accepted but ignored).
  5. Everything from v7 that was already legitimate post-hoc evidence is kept:
     terminated-for-efficacy-failure as negative, approved-in-indication as
     positive (a real regulatory fact, checked after the trial), safety- and
     accrual-driven terminations left indeterminate.

This produces far fewer determinate labels than v7 (which is the point — v7's
extra labels were fabricated from protocol boilerplate). Use labels_v8.py's
`build_labels_v8` with `fetch_live=True` to recover many of them legitimately,
from structured trial results and linked publications instead of prose.
"""

import pandas as pd
from config import (
    POSITIVE_TEXT_TRIGGERS, NEGATIVE_TEXT_TRIGGERS,
    TERMINATED_EFFICACY_REASONS, TERMINATED_SUCCESS_REASONS, log,
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

def _termination_is_success(why_stopped: str) -> bool:
    """
    Stopped early BECAUSE the drug was working (interim efficacy triggered
    early stopping) — a genuine positive outcome, and the mirror image of
    _termination_is_efficacy(). See TERMINATED_SUCCESS_REASONS in config.py.
    """
    if not why_stopped or not isinstance(why_stopped, str):
        return False
    ws = why_stopped.lower()
    return any(r in ws for r in TERMINATED_SUCCESS_REASONS)

def _has_results_posted(trial) -> bool:
    return bool(trial.get("results_first_posted") or trial.get("has_results"))

def _approved_in_indication(drug: str, disease: str) -> bool:
    approved_diseases = APPROVED_INDICATIONS.get(drug.lower(), [])
    return any(d in disease.lower() for d in approved_diseases)


def assign_labels(trial, known_approvals: set) -> dict:
    """
    Assign labels from outcome signals only — no biology, and (v9) no
    pre-trial text. Free-text evidence is read ONLY from `why_stopped`, the
    single field ClinicalTrials.gov writes after a trial's fate is known;
    `brief_summary` and `primary_outcome` are written at registration and are
    never used here (see the module docstring for why that mattered).

    LABEL PHILOSOPHY:
    - label_strict:     Only unambiguous post-hoc outcomes: an explicit
                        why_stopped statement, or regulatory approval in this
                        specific indication (a real-world fact, not text
                        matching).
    - label_balanced:   Same as strict (kept as a separate column for
                        pipeline compatibility; v9 does not add anything
                        strict lacks that is still circularity-free).
    - label_permissive: Adds nothing beyond strict/balanced by design — the
                        v7 permissive rules either duplicated strict-eligible
                        evidence or admitted pre-trial text, which is exactly
                        the bug this rewrite removes. Kept for pipeline/report
                        compatibility.

    A trial with posted results but no post-hoc text and no approval record is
    genuinely UNKNOWN from this dataset and is left indeterminate (-1) rather
    than defaulted to negative. Recovering it legitimately requires structured
    trial results or a linked publication — see labels_v8.build_labels_v8.
    """
    status = str(trial.get("status", "")).upper()
    why_stopped = str(trial.get("why_stopped", "") or "")
    canonical_drug = str(trial.get("canonical_drug", "") or "").lower()
    disease = str(trial.get("disease", "") or "").lower()

    # Post-hoc signal ONLY — computed from why_stopped alone.
    why_signal = _text_signal(why_stopped)
    approved_here = canonical_drug in known_approvals and _approved_in_indication(
        canonical_drug, disease)
    is_efficacy_stop = _termination_is_efficacy(why_stopped)
    is_success_stop = _termination_is_success(why_stopped)
    is_safety_stop = _termination_is_safety(why_stopped)
    is_accrual_stop = _termination_is_accrual(why_stopped)

    # ── STRICT: only unambiguous, post-hoc signals ────────────────────────────
    if status == "COMPLETED":
        if approved_here:
            # Regulatory approval in this exact indication is a real-world
            # outcome fact, checked after the trial — not a text match.
            strict = 1; prov = "completed+approved_indication"
        elif why_signal == "negative":
            strict = 0; prov = "completed+why_stopped_negative"
        else:
            # Results may be posted, but with no post-hoc text and no
            # approval, this trial's outcome is genuinely unknown from this
            # dataset. Do not guess.
            strict = -1; prov = "completed+unknown"

    elif status == "TERMINATED":
        if is_success_stop and not is_efficacy_stop:
            # Stopped early BECAUSE the drug was working — a real positive.
            strict = 1; prov = "terminated+early_success"
        elif is_efficacy_stop:
            strict = 0; prov = "terminated+efficacy_failure"
        elif why_signal == "negative":
            strict = 0; prov = "terminated+negative_text"
        elif is_safety_stop:
            # Toxicity, not efficacy — the drug may have worked. Unknown.
            strict = -1; prov = "terminated+safety"
        elif is_accrual_stop:
            # Logistics, not biology. Unknown.
            strict = -1; prov = "terminated+accrual"
        else:
            strict = -1; prov = "terminated+indeterminate"

    elif status == "WITHDRAWN":
        if why_signal == "negative":
            strict = 0; prov = "withdrawn+negative"
        else:
            strict = -1; prov = "withdrawn+indeterminate"

    else:
        strict = -1; prov = "other+indeterminate"

    # ── BALANCED / PERMISSIVE: identical by design (see docstring) ───────────
    balanced = strict
    permissive = strict

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