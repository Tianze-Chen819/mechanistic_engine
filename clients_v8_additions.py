"""
clients_v8_additions.py — New API functions for v8 changes.

New functions:
  - fetch_trial_results(): pulls structured results from ClinicalTrials.gov
  - fetch_pubmed_publication(): finds primary publication by NCT number
  - get_io_features(): IO-specific biology features (TMB, PD-L1, infiltration)
"""

import time
import requests
import pandas as pd
import numpy as np
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

        # Extract p-value and effect size from analyses
        analyses = po.get("analyses", [])
        for analysis in analyses:
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

POSITIVE_PUB_TERMS = [
    "met primary endpoint", "significant improvement", "superior",
    "statistically significant", "demonstrated efficacy", "positive",
    "overall survival benefit", "progression-free survival benefit",
    "objective response", "phase 3", "approved",
]
NEGATIVE_PUB_TERMS = [
    "did not meet", "failed to meet", "no significant", "no benefit",
    "futility", "negative", "did not demonstrate", "not superior",
    "no improvement", "no statistically significant",
]

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

    # Fetch abstract
    abstract_data = cached_get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={
            "db": "pubmed",
            "id": ids[0],
            "rettype": "abstract",
            "retmode": "json",
            "tool": "mech_engine",
        },
        tag=f"pubmed_abstract_{ids[0]}",
    )
    time.sleep(0.35)

    if not abstract_data:
        return result

    # Parse abstract text
    articles = abstract_data.get("PubmedArticleSet", {}).get("PubmedArticle", [])
    if isinstance(articles, dict):
        articles = [articles]

    if articles:
        article = articles[0]
        medline = article.get("MedlineCitation", {})
        article_data = medline.get("Article", {})

        # Get publication year
        pub_date = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        result["pub_year"] = pub_date.get("Year", None)

        # Get abstract text
        abstract = article_data.get("Abstract", {}).get("AbstractText", "")
        if isinstance(abstract, list):
            abstract = " ".join([a if isinstance(a, str) else a.get("#text", "") for a in abstract])

        abstract_lower = str(abstract).lower()
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