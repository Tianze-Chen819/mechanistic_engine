"""
labels_v8.py — Improved label construction using structured results + publications.

v8 CHANGES:
  1. Structured results from ClinicalTrials.gov used as primary label source
  2. PubMed publication signal used as secondary label source
  3. Distinguishes completed-no-results from completed-with-negative-results
  4. Endpoint type extracted for separate model training
  5. Continuous outcome score computed where effect size is available
  6. IO drug flag added so IO/non-IO models can be trained separately

v9 FIX: the priority-3 text fallback (used whenever a trial has neither a
structured p-value nor a linked publication — most of the dataset, since the
majority of single-arm Phase 2 oncology trials report a descriptive response
rate with no formal hypothesis test) previously matched trigger phrases
against `why_stopped + brief_summary + primary_outcome` combined. The last two
are written at trial registration, before the outcome is known — see
docs/label_audit.md, which found 100% of v7's positive labels fired only on
those two pre-trial fields. The fallback now reads only `why_stopped`, and a
COMPLETED trial with no post-hoc text and no approval record is left
indeterminate rather than defaulted to negative.
"""

import pandas as pd
import numpy as np
from config import (
    POSITIVE_TEXT_TRIGGERS, NEGATIVE_TEXT_TRIGGERS,
    TERMINATED_EFFICACY_REASONS, log,
)
from clients_v8_additions import fetch_all_trial_results, fetch_all_publications

# IO drug targets — used to flag trials for separate IO model
IO_TARGETS = {"PDCD1", "CD274", "CTLA4"}

# Approved indications (same as labels.py)
from labels import (
    APPROVED_INDICATIONS, _text_signal, _termination_is_efficacy,
    _termination_is_success, _has_results_posted,
)

def _approved_in_indication(drug: str, disease: str) -> bool:
    approved = APPROVED_INDICATIONS.get(drug.lower(), [])
    return any(d in disease.lower() for d in approved)


def assign_labels_v8(trial, structured_results: dict, publication: dict,
                      known_approvals: set) -> dict:
    """
    v8 label assignment using all available evidence sources in priority order:
    1. Structured results p-value (most reliable — trial-specific)
    2. Publication endpoint signal (reliable — peer-reviewed)
    3. Trial record text signal (less reliable — often incomplete)
    4. FDA approval in specific indication (drug-indication level, not trial-specific)
    5. Results posted but no positive signal (weak negative)
    """
    status = str(trial.get("status", "")).upper()
    why_stopped = str(trial.get("why_stopped", "") or "")
    canonical_drug = str(trial.get("canonical_drug", "") or "").lower()
    disease = str(trial.get("disease", "") or "").lower()
    modality = str(trial.get("modality", "") or "").lower()
    target = str(trial.get("primary_target", "UNKNOWN"))

    # v9: post-hoc text signal ONLY — why_stopped, never brief_summary or
    # primary_outcome (both are written before the trial's outcome exists).
    why_signal = _text_signal(why_stopped)
    approved_here = _approved_in_indication(canonical_drug, disease)
    is_io_trial = target in IO_TARGETS or modality in ("antibody",) and target in {"PDCD1", "CD274", "CTLA4"}

    # Structured results (most reliable source)
    sr = structured_results or {}
    has_sr = sr.get("has_structured_results", False)
    sr_met = sr.get("primary_endpoint_met", None)
    sr_pval = sr.get("primary_p_value", None)
    sr_effect = sr.get("primary_effect_size", None)
    endpoint_type = sr.get("endpoint_type", None)

    # Publication signal (second most reliable)
    pub = publication or {}
    has_pub = pub.get("has_publication", False)
    pub_met = pub.get("pub_endpoint_met", None)
    pub_signal = pub.get("pub_text_signal", "neutral")

    # ── Determine label ───────────────────────────────────────────────────────

    # Priority 1: structured results p-value
    if has_sr and sr_met is not None:
        strict = 1 if sr_met else 0
        prov = f"structured_results_pval_{sr_pval:.3f}" if sr_pval else "structured_results"

    # Priority 2: publication endpoint signal
    elif has_pub and pub_met is not None:
        strict = 1 if pub_met else 0
        prov = f"publication_{pub_signal}"

    # Priority 3: why_stopped text / regulatory approval (v9: post-hoc only —
    # no structured p-value and no linked publication were found, and the
    # ONLY remaining evidence this trial can offer is what happened after it
    # ended, i.e. why_stopped, or whether the drug is now approved here).
    elif status == "COMPLETED":
        if approved_here:
            strict = 1; prov = "completed+approved_indication"
        elif why_signal == "negative":
            strict = 0; prov = "completed+why_stopped_negative"
        else:
            # Posted results with no post-hoc text and no approval record —
            # genuinely unknown from this dataset. Do not guess from
            # brief_summary/primary_outcome (see module docstring).
            strict = -1; prov = "completed+unknown"

    elif status == "TERMINATED":
        is_efficacy_stop = _termination_is_efficacy(why_stopped)
        is_success_stop = _termination_is_success(why_stopped)
        if is_success_stop and not is_efficacy_stop:
            strict = 1; prov = "terminated+early_success"
        elif is_efficacy_stop:
            strict = 0; prov = "terminated+efficacy_failure"
        elif why_signal == "negative":
            strict = 0; prov = "terminated+negative_text"
        else:
            strict = -1; prov = "terminated+indeterminate"

    elif status == "WITHDRAWN":
        if why_signal == "negative":
            strict = 0; prov = "withdrawn+negative"
        else:
            strict = -1; prov = "withdrawn+indeterminate"

    else:
        strict = -1; prov = "other+indeterminate"

    # v9: permissive == strict. approved_here is now checked as priority 3's
    # first COMPLETED rule (above), so by the time strict is -1 for a
    # COMPLETED trial, approved_here is already known False — there is
    # nothing left for a separate "permissive" pass to add without
    # reintroducing pre-trial text as evidence.
    permissive = strict

    # Continuous outcome score (for regression model)
    # Use effect size where available, otherwise binary
    if sr_effect is not None and endpoint_type == "response":
        # Response rate: 0-100% → 0-1
        continuous_score = min(sr_effect / 100.0, 1.0) if sr_effect > 1 else sr_effect
    elif sr_pval is not None:
        # Convert p-value to confidence: lower p = higher score
        continuous_score = max(0, 1 - (sr_pval * 20))  # p=0.05 → score=0
    elif strict == 1:
        continuous_score = 0.75  # positive but no effect size
    elif strict == 0:
        continuous_score = 0.25  # negative but no effect size
    else:
        continuous_score = None  # indeterminate

    return {
        "label_strict": strict,
        "label_balanced": strict,
        "label_permissive": permissive,
        "label_provenance": prov,
        "endpoint_type": endpoint_type,
        "continuous_outcome": continuous_score,
        "has_structured_results": has_sr,
        "has_publication": has_pub,
        "is_io_trial": 1 if is_io_trial else 0,
        "effect_size": sr_effect,
    }


def build_labels_v8(df: pd.DataFrame, fetch_live: bool = True) -> pd.DataFrame:
    """
    Build v8 labels with structured results and publication signals.
    
    fetch_live=True: fetch from APIs (slow first run, cached after)
    fetch_live=False: use text signals only (fast, for testing)
    """
    known_approvals = set(APPROVED_INDICATIONS.keys())
    nct_ids = df["nct_id"].tolist()

    if fetch_live:
        # Fetch structured results and publications
        structured_results = fetch_all_trial_results(nct_ids)
        publications = fetch_all_publications(nct_ids)
    else:
        structured_results = {nct: {} for nct in nct_ids}
        publications = {nct: {} for nct in nct_ids}

    records = []
    for _, trial in df.iterrows():
        nct = trial.get("nct_id", "")
        sr = structured_results.get(nct, {})
        pub = publications.get(nct, {})
        labels = assign_labels_v8(trial, sr, pub, known_approvals)
        records.append(labels)

    label_df = pd.DataFrame(records, index=df.index)
    result = pd.concat([df, label_df], axis=1)

    # Summary
    for lc in ["label_strict", "label_permissive"]:
        pos = (result[lc] == 1).sum()
        neg = (result[lc] == 0).sum()
        ind = (result[lc] == -1).sum()
        total = pos + neg
        rate = pos / total if total > 0 else 0
        print(f"\n  {lc}: {pos} pos ({rate:.1%}), {neg} neg, {ind} indeterminate")

    # Endpoint type distribution
    ep_dist = result[result["endpoint_type"].notna()]["endpoint_type"].value_counts()
    print(f"\n  Endpoint types: {ep_dist.to_dict()}")

    # IO trial count
    io_count = result["is_io_trial"].sum()
    print(f"  IO trials flagged: {io_count}")

    # Continuous outcome coverage
    cont_coverage = result["continuous_outcome"].notna().sum()
    print(f"  Continuous outcome scores: {cont_coverage}/{len(result)}")

    has_sr = result["has_structured_results"].sum()
    has_pub = result["has_publication"].sum()
    print(f"  Structured results: {has_sr}/{len(result)}")
    print(f"  Publication links: {has_pub}/{len(result)}")

    log.info(f"v8 Labels built: SR={has_sr}, pub={has_pub}, io={io_count}")
    return result