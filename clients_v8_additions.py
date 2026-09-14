"""
clients_v8_additions.py — New API functions for v8 changes.

New functions:
  - fetch_trial_results(): pulls structured results from ClinicalTrials.gov
  - fetch_pubmed_publication(): finds primary publication by NCT number
  - get_io_features(): IO-specific biology features (TMB, PD-L1, infiltration)
"""

import re
import time
import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from config import CACHE_DIR, USE_CACHE, CACHE_TTL_DAYS, log
import json
from datetime import datetime, timedelta

# Local cache helpers (duplicated from clients.py to avoid circular import)
def _cache_key_v8(tag: str):
    safe = tag.replace("/", "_").replace("?", "_").replace("&", "_")[:120]
    return CACHE_DIR / f"{safe}.json"

def _is_fresh_v8(path) -> bool:
    if not USE_CACHE or not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(days=CACHE_TTL_DAYS)

SESSION_V8 = requests.Session()
SESSION_V8.headers.update({"User-Agent": "MechanisticEngine/8.0 (research)"})

def cached_get(url: str, params: dict = None, tag: str = None,
               timeout: int = 30, retries: int = 3):
    cache_tag = tag or (url + str(sorted((params or {}).items())))
    cpath = _cache_key_v8(cache_tag)
    if _is_fresh_v8(cpath):
        try:
            return json.loads(cpath.read_text())
        except Exception:
            pass
    for attempt in range(retries):
        try:
            r = SESSION_V8.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            try:
                cpath.write_text(json.dumps(data, default=str))
            except Exception:
                pass
            return data
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.warning(f"GET attempt {attempt+1}: {e}")
            time.sleep(min(2**attempt, 4))
    return None

def cached_get_xml(url: str, params: dict, tag: str,
                   timeout: int = 30, retries: int = 3) -> str | None:
    """
    GET expecting XML back, with disk caching of the raw text.

    v9 ADDED. PubMed's efetch endpoint claims `retmode=json` and returns a
    `Content-Type: application/json` header, but the body is actually plain
    text — `r.json()` on it always raises. The original code used cached_get()
    for this call, so every request silently failed after 3 retries with
    exponential backoff (up to ~7s wasted) and `fetch_pubmed_by_nct` returned
    empty results for every trial it was ever called on. efetch's `retmode=xml`
    is the one that actually returns structured data.
    """
    cpath = _cache_key_v8(tag).with_suffix(".xml")
    if _is_fresh_v8(cpath):
        try:
            return cpath.read_text()
        except Exception:
            pass
    for attempt in range(retries):
        try:
            r = SESSION_V8.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            text = r.text
            try:
                cpath.write_text(text)
            except Exception:
                pass
            return text
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.warning(f"XML GET attempt {attempt+1}: {e}")
            time.sleep(min(2**attempt, 4))
    return None


def cached_post(url: str, payload: dict, tag: str,
                timeout: int = 30, retries: int = 3):
    cpath = _cache_key_v8(tag)
    if _is_fresh_v8(cpath):
        try:
            return json.loads(cpath.read_text())
        except Exception:
            pass
    for attempt in range(retries):
        try:
            r = SESSION_V8.post(url, json=payload,
                                headers={"Content-Type": "application/json"},
                                timeout=timeout)
            r.raise_for_status()
            data = r.json()
            try:
                cpath.write_text(json.dumps(data, default=str))
            except Exception:
                pass
            return data
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.warning(f"POST attempt {attempt+1}: {e}")
            time.sleep(min(2**attempt, 4))
    return None

# ── 1. Structured results from ClinicalTrials.gov ─────────────────────────────

def fetch_trial_results(nct_id: str) -> dict:
    """
    Fetch structured results section from ClinicalTrials.gov v2 API.
    Returns primary endpoint outcome, p-value, and effect size where available.
    """
    data = cached_get(
        f"https://clinicaltrials.gov/api/v2/studies/{nct_id}",
        params={"fields": "ResultsSection,StatusModule"},
        tag=f"ct_results_{nct_id}",
        timeout=20,
    )

    result = {
        "has_structured_results": False,
        "primary_endpoint_met": None,      # True/False/None
        "primary_p_value": None,
        "primary_effect_size": None,       # response rate, HR, etc.
        "endpoint_type": None,             # survival/response/biomarker/pfs
        "result_text": "",
    }

    if not data:
        return result

    results_section = data.get("protocolSection", {})
    results_data = data.get("resultsSection", {})

    if not results_data:
        return result

    result["has_structured_results"] = True

    # Parse primary outcome measures
    outcomes = results_data.get("outcomeMeasuresModule", {}).get("outcomeMeasures", [])
    primary_outcomes = [o for o in outcomes if o.get("type", "").upper() == "PRIMARY"]

    if primary_outcomes:
        po = primary_outcomes[0]
        title = po.get("title", "").lower()
        description = po.get("description", "").lower()

        # Classify endpoint type
        if any(kw in title for kw in ["overall survival", "os ", "death"]):
            result["endpoint_type"] = "survival"
        elif any(kw in title for kw in ["progression", "pfs", "progression-free"]):
            result["endpoint_type"] = "pfs"
        elif any(kw in title for kw in ["response", "orr", "objective response", "complete response"]):
            result["endpoint_type"] = "response"
        elif any(kw in title for kw in ["biomarker", "expression", "mutation"]):
            result["endpoint_type"] = "biomarker"
        else:
            result["endpoint_type"] = "other"

        # Extract p-value and effect size from analyses. v9: skip
        # non-inferiority analyses — "significant" there means "significantly
        # non-inferior to the margin", not "beat control", so reading it with
        # the same p<0.05-means-met rule as a superiority test would be wrong.
        analyses = po.get("analyses", [])
        for analysis in analyses:
            if analysis.get("testedNonInferiority"):
                continue
            pval = analysis.get("pValue", "")
            if pval:
                try:
                    result["primary_p_value"] = float(str(pval).replace("<", "").replace(">", "").strip())
                    result["primary_endpoint_met"] = result["primary_p_value"] < 0.05
                except (ValueError, TypeError):
                    pass

            # Extract effect size (HR, OR, RR)
            param_value = analysis.get("paramValue", "")
            if param_value:
                try:
                    result["primary_effect_size"] = float(param_value)
                except (ValueError, TypeError):
                    pass

        # Fallback: check dispersion / groups for response rate
        groups = po.get("groups", [])
        for group in groups:
            measurements = group.get("measurements", [])
            for m in measurements:
                val = m.get("value", "")
                try:
                    result["primary_effect_size"] = float(val)
                except (ValueError, TypeError):
                    pass

    # Parse statistical analyses text for endpoint met signal
    baseline = results_data.get("baselineCharacteristicsModule", {})
    adverse = results_data.get("adverseEventsModule", {})

    return result


def fetch_all_trial_results(nct_ids: list[str]) -> dict[str, dict]:
    """Fetch structured results for all trials. Returns dict keyed by NCT ID."""
    from tqdm import tqdm
    results = {}
    print(f"  Fetching structured results for {len(nct_ids)} trials...")
    for nct_id in tqdm(nct_ids, desc="CT.gov results"):
        results[nct_id] = fetch_trial_results(nct_id)
        time.sleep(0.1)  # be polite to the API
    
    has_results = sum(1 for r in results.values() if r["has_structured_results"])
    has_pval = sum(1 for r in results.values() if r["primary_p_value"] is not None)
    print(f"  Structured results: {has_results}/{len(nct_ids)} trials")
    print(f"  With p-value: {has_pval}/{len(nct_ids)} trials")
    return results


# ── 2. PubMed publication linking by NCT number ───────────────────────────────


# v9 FIX: bare "positive"/"negative"/"approved"/"superior" were removed. Unlike
# brief_summary/primary_outcome, an abstract IS post-hoc evidence — but these
# specific words are still too generic: "superior" appears in comparator-drug
# background, and "approved" in an abstract routinely refers to a DIFFERENT,
# already-approved drug used as the control arm, not this trial's drug. Every
# phrase below only makes sense as a statement about how THIS trial's own
# primary endpoint came out.
POSITIVE_PUB_TERMS = [
    "met the primary endpoint", "met its primary endpoint",
    "significant improvement in", "statistically significant improvement",
    "demonstrated efficacy", "significantly prolonged",
    "significantly improved overall survival",
    "significantly improved progression-free survival",
    "achieved the primary endpoint",
]
NEGATIVE_PUB_TERMS = [
    "did not meet", "failed to meet", "no significant difference",
    "no significant improvement", "no benefit was observed", "futility",
    "did not demonstrate efficacy", "not superior to", "no improvement in",
    "no statistically significant difference", "failed to demonstrate",
]

# v9 ADDED. Oncology RCT abstracts almost always report the trial's own
# p-value in the conclusion ("...median OS was 13.7 vs 6.5 months (P = .006)").
# That is a much stronger post-hoc signal than matching a fixed phrase list —
# it is the same p<0.05 threshold already used for structured results, just
# sourced from the published text instead of the CT.gov results section, for
# the (common) trials that report to a journal without ever posting structured
# results to the registry.
PVALUE_RE = re.compile(r"\bp\s*(?:-?value)?\s*[<=]\s*(\.\d+|\d+\.\d+|\d+)", re.IGNORECASE)
# A null-result phrase in the same sentence as the smallest p-value vetoes a
# positive read (see _pvalue_signal).
NULL_WORDS = ("no significant", "did not", "no differ", "not superior",
             "no benefit", "failed to")


def _pvalue_signal(abstract: str) -> bool | None:
    """
    True/False from the smallest reported p-value in the abstract; None if no
    usable p-value is found. A sentence containing an explicit null-result
    phrase ("no significant difference, P = .34") vetoes a positive read even
    if the number itself would otherwise look significant — the abstract is
    reporting a comparison that did NOT favour the drug. Otherwise p<0.05 is
    read as the endpoint being met: BENEFIT_WORDS in the same sentence
    corroborate this but are not required, since sponsors overwhelmingly
    report the p-value that supports their hypothesis in the first place.
    """
    sentences = re.split(r"(?<=[.!?])\s+", abstract)
    best = None  # (pvalue, sentence)
    for s in sentences:
        for m in PVALUE_RE.finditer(s):
            raw = m.group(1)
            try:
                # ".006" must parse as 0.006, not 6 — float(".006") does this
                # correctly already, but a bare "006" (no dot) would parse as
                # 6.0, which the regex's own alternatives prevent it matching
                # unless the source text itself omitted the decimal point.
                p = float(raw)
            except ValueError:
                continue
            if best is None or p < best[0]:
                best = (p, s.lower())
    if best is None:
        return None
    p, sentence = best
    if any(w in sentence for w in NULL_WORDS):
        return False
    return p < 0.05


def fetch_pubmed_by_nct(nct_id: str) -> dict:
    """
    Search PubMed for primary publication of a trial by NCT number.
    Returns publication signal and abstract text.
    """
    result = {
        "has_publication": False,
        "pub_endpoint_met": None,
        "pub_text_signal": "neutral",
        "pubmed_id": None,
        "pub_year": None,
    }

    # Search PubMed for this NCT ID
    search_data = cached_get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": f"{nct_id}[si] OR {nct_id}[Title/Abstract]",
            "retmax": 5,
            "retmode": "json",
            "sort": "relevance",
            "tool": "mech_engine",
        },
        tag=f"pubmed_nct_{nct_id}",
    )
    time.sleep(0.35)  # NCBI rate limit

    if not search_data:
        return result

    ids = search_data.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return result

    result["has_publication"] = True
    result["pubmed_id"] = ids[0]

    # Fetch abstract. v9: retmode=xml, not json — see cached_get_xml docstring.
    xml_text = cached_get_xml(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={
            "db": "pubmed",
            "id": ids[0],
            "rettype": "abstract",
            "retmode": "xml",
            "tool": "mech_engine",
        },
        tag=f"pubmed_abstract_{ids[0]}",
    )
    time.sleep(0.35)

    if not xml_text:
        return result

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return result

    article = root.find(".//PubmedArticle")
    if article is None:
        return result

    year_el = article.find(".//Journal/JournalIssue/PubDate/Year")
    if year_el is not None and year_el.text:
        result["pub_year"] = year_el.text
    else:
        medline_date = article.find(".//Journal/JournalIssue/PubDate/MedlineDate")
        if medline_date is not None and medline_date.text:
            result["pub_year"] = medline_date.text[:4]

    abstract = " ".join(
        (el.text or "") for el in article.findall(".//Abstract/AbstractText")
    )
    abstract_lower = abstract.lower()

    # Primary: the abstract's own reported p-value (see _pvalue_signal).
    pval_met = _pvalue_signal(abstract)
    if pval_met is not None:
        result["pub_endpoint_met"] = pval_met
        result["pub_text_signal"] = "positive" if pval_met else "negative"
        return result

    # Fallback: curated phrases, for abstracts that report an outcome without
    # a bare "P = " value (e.g. "did not reach statistical significance").
    neg = sum(1 for t in NEGATIVE_PUB_TERMS if t in abstract_lower)
    pos = sum(1 for t in POSITIVE_PUB_TERMS if t in abstract_lower)
    if neg > 0 and neg >= pos:
        result["pub_text_signal"] = "negative"
        result["pub_endpoint_met"] = False
    elif pos > 0:
        result["pub_text_signal"] = "positive"
        result["pub_endpoint_met"] = True
    else:
        result["pub_text_signal"] = "neutral"

    return result


def fetch_all_publications(nct_ids: list[str]) -> dict[str, dict]:
    """Fetch PubMed publications for all trials."""
    from tqdm import tqdm
    results = {}
    print(f"  Linking {len(nct_ids)} trials to PubMed publications...")
    for nct_id in tqdm(nct_ids, desc="PubMed NCT link"):
        results[nct_id] = fetch_pubmed_by_nct(nct_id)

    has_pub = sum(1 for r in results.values() if r["has_publication"])
    has_signal = sum(1 for r in results.values() if r["pub_endpoint_met"] is not None)
    print(f"  Publications found: {has_pub}/{len(nct_ids)}")
    print(f"  With clear endpoint signal: {has_signal}/{len(nct_ids)}")
    return results


# ── 3. IO-specific biology features ──────────────────────────────────────────

# Curated IO-relevant biology scores per target
# Sources: published TMB/PD-L1 association literature, TCGA immune subtype data
IO_FEATURES = {
    # (target): (tmb_relevance, pdl1_expression, immune_infiltration, io_mechanism)
    # tmb_relevance: how much TMB predicts response for drugs targeting this
    # pdl1_expression: PD-L1 expression level on tumour cells (0-1)
    # immune_infiltration: immune cell density in tumours (0-1)
    # io_mechanism: 1 if drug directly targets immune checkpoint, 0 otherwise

    "PDCD1":   (0.85, 0.80, 0.85, 1.0),  # PD-1 — directly IO
    "CD274":   (0.85, 0.95, 0.80, 1.0),  # PD-L1 — directly IO
    "CTLA4":   (0.70, 0.60, 0.80, 1.0),  # CTLA-4 — directly IO
    "EGFR":    (0.20, 0.30, 0.40, 0.0),  # targeted, TMB low relevance
    "ERBB2":   (0.15, 0.35, 0.45, 0.0),
    "BRAF":    (0.40, 0.40, 0.55, 0.0),  # some TMB correlation in melanoma
    "KRAS":    (0.45, 0.45, 0.60, 0.0),
    "PIK3CA":  (0.20, 0.30, 0.40, 0.0),
    "MTOR":    (0.15, 0.25, 0.35, 0.0),
    "CDK4":    (0.20, 0.30, 0.40, 0.0),
    "BCL2":    (0.25, 0.40, 0.55, 0.0),
    "PARP1":   (0.60, 0.35, 0.50, 0.0),  # BRCA/HRD link to TMB
    "AR":      (0.10, 0.20, 0.30, 0.0),
    "ESR1":    (0.10, 0.25, 0.35, 0.0),
    "BTK":     (0.30, 0.40, 0.65, 0.0),  # B-cell
    "ABL1":    (0.20, 0.25, 0.35, 0.0),
    "ALK":     (0.15, 0.30, 0.40, 0.0),
    "MET":     (0.25, 0.35, 0.45, 0.0),
    "VEGFA":   (0.20, 0.30, 0.50, 0.0),  # anti-angiogenic affects immune TME
    "KDR":     (0.20, 0.30, 0.50, 0.0),
    "MS4A1":   (0.25, 0.35, 0.65, 0.0),  # B-cell
    "JAK1":    (0.35, 0.50, 0.60, 0.2),  # JAK-STAT affects immune signalling
    "MAP2K1":  (0.30, 0.35, 0.45, 0.0),
    "NTRK1":   (0.20, 0.25, 0.35, 0.0),
    "SMO":     (0.15, 0.20, 0.30, 0.0),
    "CYP17A1": (0.10, 0.15, 0.25, 0.0),
    "CYP19A1": (0.10, 0.15, 0.25, 0.0),
    "PSMB5":   (0.25, 0.30, 0.45, 0.0),
    "CRBN":    (0.30, 0.40, 0.55, 0.1),  # IMiDs have immune effects
    "PIK3CD":  (0.30, 0.40, 0.60, 0.1),
    "TACSTD2": (0.20, 0.35, 0.45, 0.0),
    "NECTIN4": (0.20, 0.30, 0.40, 0.0),
    "TNFRSF8": (0.35, 0.45, 0.70, 0.2),  # CD30 — lymphoma immune context
    "TYMS":    (0.15, 0.20, 0.30, 0.0),
    "TOP1":    (0.20, 0.25, 0.35, 0.0),
    "TOP2A":   (0.20, 0.25, 0.35, 0.0),
}

def get_io_features(symbol: str) -> dict:
    """Return IO-specific biology features for a target."""
    defaults = (0.2, 0.3, 0.4, 0.0)
    tmb, pdl1, infiltration, io_mech = IO_FEATURES.get(symbol, defaults)
    return {
        "tmb_relevance": tmb,
        "pdl1_expression": pdl1,
        "immune_infiltration": infiltration,
        "io_mechanism": io_mech,
        "is_io_target": 1.0 if io_mech > 0.5 else 0.0,
    }


# ── 4. Drug class classifier for TDS split ───────────────────────────────────

BIOLOGIC_TARGETS = {
    # Targets primarily addressed by biologics (antibodies, ADCs, cell therapy)
    "PDCD1", "CD274", "CTLA4", "ERBB2", "MS4A1", "VEGFA",
    "NECTIN4", "TACSTD2", "TNFRSF8",
}

def get_drug_modality_class(symbol: str, modality: str) -> str:
    """
    Classify drug as small_molecule or biologic for TDS splitting.
    Returns 'biologic' or 'small_molecule'.
    """
    biologic_modalities = {"antibody", "adc", "bispecific", "cell_therapy",
                           "gene_therapy", "peptide"}
    if modality.lower() in biologic_modalities:
        return "biologic"
    if symbol in BIOLOGIC_TARGETS:
        return "biologic"
    return "small_molecule"


def compute_split_tds(symbol: str, modality: str, features: dict) -> dict:
    """
    Compute separate TDS scores for small molecule vs biologic targets.
    
    Small molecule TDS: binding evidence, potency, tractability (ChEMBL-based)
    Biologic TDS: surface accessibility (HPA), expression level, IO mechanism
    
    This fixes the bias where antibody targets like PD-1 score low on
    small-molecule druggability metrics despite being highly druggable as
    antibody targets.
    """
    drug_class = get_drug_modality_class(symbol, modality)
    io_feats = get_io_features(symbol)

    if drug_class == "small_molecule":
        # Original small-molecule druggability metrics
        sm_tds = (
            0.30 * features.get("tractability", 0) +
            0.25 * features.get("binding_evidence", 0) +
            0.25 * features.get("potency_proxy", 0) +
            0.20 * features.get("known_drug_count", 0)
        )
        bio_tds = 0.0  # not applicable
    else:
        # Biologic druggability: surface accessibility, expression, IO mechanism
        # Surface accessibility proxied by tumor_expression (HPA)
        # Internalization proxied by network_degree (interacting proteins)
        bio_tds = (
            0.35 * features.get("tumor_expression", 0.4) +       # expression on target cells
            0.25 * io_feats["pdl1_expression"] +                  # relevant surface marker
            0.20 * (1 - features.get("normal_tissue_burden", 0.5)) +  # selectivity
            0.20 * io_feats["io_mechanism"]                        # IO relevance
        )
        sm_tds = 0.0  # not applicable for biologics

    return {
        "tds_small_molecule": sm_tds,
        "tds_biologic": bio_tds,
        "tds_combined": sm_tds if drug_class == "small_molecule" else bio_tds,
        "drug_class": 1.0 if drug_class == "biologic" else 0.0,
    }