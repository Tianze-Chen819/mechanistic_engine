"""
auto_target_mapping.py — Automated drug->target discovery via ChEMBL, as a
fallback for drugs the curated DRUG_TARGET_DB doesn't know about.

WHY THIS EXISTS

drug_target_db.py hand-curates ~100 drugs. In the full ClinicalTrials.gov
oncology corpus (thousands of interventions — many single-site academic
trials of less common agents, older chemotherapies, or newly named
investigational compounds), a fixed hand-typed dictionary leaves a large
fraction of trials "unmapped" — every unmapped trial's biology features are
then all zero, which both shrinks the usable dataset and skews it toward
whatever a curator happened to already know about (mostly blockbuster,
already-approved drugs). That skew is one of this project's known
limitations (see docs/model_limitations.md): the corpus over-represents
already-validated mechanisms, which caps how much variance a biology score
can explain.

Hand-typing MORE drug/target pairs from memory would not fix this safely —
it would just add more entries that are only as reliable as the curator's
recollection, with no way for a reader to check where they came from. This
module instead queries ChEMBL's own structured mechanism-of-action database
(https://www.ebi.ac.uk/chembl/), which is exactly the kind of "documented
public API, deterministic and auditable" source config.py's stated principles
call for:

    drug name --[ChEMBL molecule search]--> ChEMBL ID
             --[ChEMBL mechanism]--> target ChEMBL ID + action type
             --[ChEMBL target]--> UniProt accession + HGNC gene symbol

Every result is cached to disk with its full ChEMBL provenance
(molecule_chembl_id, target_chembl_id, mechanism_of_action text) so it can be
audited or corrected later — this is discovery, not invention.

LIMITATION, STATED PLAINLY: a drug with multiple ChEMBL mechanism records
(e.g. a multi-kinase inhibitor) has this module pick the FIRST one ChEMBL
returns as "primary_target". This is the same simplification the rest of the
pipeline already makes for hand-curated multi-target drugs — it is not solved
here, just extended to auto-discovered drugs with the same caveat.

USAGE
    from auto_target_mapping import chembl_lookup, build_auto_mappings
    entry = chembl_lookup("repotrectinib")
    # {"targets": ["ROS1"], "modality": "small_molecule", "moa": "chembl_auto",
    #  "confidence": 0.6, "source": "chembl", "chembl_id": "CHEMBL4298138", ...}
"""

import json
import time
from pathlib import Path

import requests

from config import CACHE_DIR, log

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
_CACHE_PATH = CACHE_DIR / "chembl_auto_target_map.json"
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "MechanisticEngine-autotarget/1.0"})


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    try:
        _CACHE_PATH.write_text(json.dumps(cache, indent=1, default=str))
    except Exception as e:
        log.warning(f"Could not save ChEMBL auto-map cache: {e}")


def _get(url: str, params: dict, retries: int = 3):
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            time.sleep(min(2 ** attempt, 4))
        except Exception:
            time.sleep(min(2 ** attempt, 4))
    return None


def _looks_like_a_drug(name: str) -> bool:
    """Skip strings that are clearly not a single drug name — querying
    ChEMBL for "placebo" or a three-drug combination string wastes a call
    and can't return a meaningful single target anyway."""
    n = name.lower().strip()
    if not n or len(n) < 3:
        return False
    junk = ("placebo", "best supportive care", "standard of care", "observation",
           "no intervention", "sham", "matching placebo")
    if any(j in n for j in junk):
        return False
    if "+" in n or " plus " in n or "/" in n or " and " in n:
        return False  # combination regimen string, not a single agent
    return True


def chembl_lookup(drug_name: str, cache: dict | None = None) -> dict | None:
    """
    Look up a single drug's primary mechanism target via ChEMBL. Returns None
    if ChEMBL has no molecule match or no recorded mechanism (most likely for
    non-drug strings, or genuinely novel compounds with no public MoA data
    yet — NOT a reason to guess).
    """
    key = drug_name.lower().strip()
    own_cache = cache if cache is not None else _load_cache()
    if key in own_cache:
        return own_cache[key]

    if not _looks_like_a_drug(key):
        own_cache[key] = None
        if cache is None:
            _save_cache(own_cache)
        return None

    mol_data = _get(f"{CHEMBL_API}/molecule/search", {"q": key, "format": "json"})
    molecules = (mol_data or {}).get("molecules", [])
    result = None
    if molecules:
        chembl_id = molecules[0]["molecule_chembl_id"]
        mech_data = _get(f"{CHEMBL_API}/mechanism",
                         {"molecule_chembl_id": chembl_id, "format": "json"})
        mechanisms = (mech_data or {}).get("mechanisms", [])
        if mechanisms:
            m = mechanisms[0]  # see LIMITATION in module docstring
            target_id = m.get("target_chembl_id")
            gene_symbol = None
            if target_id:
                tgt_data = _get(f"{CHEMBL_API}/target/{target_id}", {"format": "json"})
                for comp in (tgt_data or {}).get("target_components", []):
                    for syn in comp.get("target_component_synonyms", []):
                        if syn.get("syn_type") == "GENE_SYMBOL":
                            gene_symbol = syn.get("component_synonym")
                            break
                    if gene_symbol:
                        break
            if gene_symbol:
                action = (m.get("action_type") or "").lower()
                modality = "antibody" if "antibody" in (m.get("mechanism_of_action") or "").lower() \
                    else "small_molecule"
                result = {
                    "targets": [gene_symbol], "modality": modality,
                    "moa": "chembl_auto", "confidence": 0.6, "source": "chembl",
                    "chembl_molecule_id": chembl_id, "chembl_target_id": target_id,
                    "mechanism_of_action": m.get("mechanism_of_action"),
                    "action_type": action,
                }

    own_cache[key] = result
    if cache is None:
        _save_cache(own_cache)
    time.sleep(0.2)  # be polite to the EBI API
    return result


def build_auto_mappings(unmapped_drug_names: list[str]) -> dict:
    """
    Batch version — looks up every name, sharing one cache write at the end.
    Returns {drug_name_lower: entry} for names ChEMBL could actually resolve;
    unresolved names are simply absent (still 'unmapped' downstream, but the
    fact that ChEMBL was CHECKED and returned nothing is itself informative
    and is recorded in the cache file for auditing).
    """
    cache = _load_cache()
    found = {}
    checked = 0
    for name in unmapped_drug_names:
        key = name.lower().strip()
        if key in cache and cache[key] is not None:
            found[key] = cache[key]
            continue
        entry = chembl_lookup(name, cache=cache)
        checked += 1
        if entry:
            found[key] = entry
    _save_cache(cache)
    log.info(f"ChEMBL auto-mapping: checked {checked} new names, "
             f"resolved {len(found)} to a target")
    return found


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or ["repotrectinib", "capmatinib", "tepotinib",
                             "mobocertinib", "sacituzumab govitecan",
                             "placebo", "unknown drug xyz123"]
    print(f"Testing ChEMBL auto-lookup on {len(names)} names...\n")
    for n in names:
        entry = chembl_lookup(n)
        if entry:
            print(f"  {n:<28} -> {entry['targets'][0]:<10} "
                  f"({entry['action_type']}, conf={entry['confidence']}) "
                  f"[{entry['chembl_molecule_id']}]")
        else:
            print(f"  {n:<28} -> not resolved")
