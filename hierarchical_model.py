"""
hierarchical_model.py — Empirical-Bayes shrinkage for small-sample biology profiles.

WHY THIS EXISTS (see docs/label_audit.md and docs/model_limitations.md)

The audit found that 638 labelled trials collapse onto ~200 distinct
(target, disease) biology profiles across 34 targets, and some of the most
mechanistically interesting pairs — a novel target in a rare cancer type —
have only 1-3 trials. A plain success-rate estimate for a pair with 2 trials
(0 or 1 positive) is not a stable number: it is either 0%, 50%, or 100%, and
none of those is trustworthy. Feeding features from that pair into a
supervised model doesn't fix this — the model still has only those 1-3 rows
to learn the pair's contribution from.

The standard fix for "the group I care about is too small to estimate on its
own" is EMPIRICAL BAYES SHRINKAGE: borrow statistical strength from other,
similar groups. A pair's estimate is pulled toward the average of the
mechanism family it belongs to (e.g. other immune-checkpoint targets, other
angiogenesis targets), and that family average is in turn pulled toward the
disease-wide average, and that toward the global rate. A pair with only 2
trials ends up mostly reflecting its family's rate (wide credible interval,
honestly flagged as uncertain); a pair with 40 trials ends up reflecting
almost entirely its own data (narrow interval). This is the same technique
used for small-sample rate estimation in sports analytics and epidemiology
(a Beta-Binomial hierarchical model fit by method of moments — no new
dependencies required).

This directly produces a DECISION-SUPPORT number that plain point-estimates
and plain ML probabilities do not: an interval that widens honestly when data
is thin, instead of reporting 100% success from 1 lucky trial or 0% from 1
unlucky one.

USAGE
    from hierarchical_model import fit_hierarchical_rates, mechanism_family
    rates = fit_hierarchical_rates(df, label_col="label_permissive")
    # rates indexed by (target, disease): posterior_mean, ci_lo, ci_hi, n, k,
    # family, shrinkage_weight (0 = fully borrowed, 1 = fully own-data)
"""

import numpy as np
import pandas as pd
from scipy import stats

# ── Mechanism family grouping ──────────────────────────────────────────────
#
# A coarse grouping by known pharmacology, used ONLY to decide which pairs
# should lend each other statistical strength — not used as a model feature
# (that would just be a weaker version of the raw biology features already
# in the pipeline). Targets not listed fall back to a modality-based bucket
# so new targets in a larger corpus degrade gracefully instead of forming
# singleton, unhelpful "families".
TARGET_FAMILY = {
    "PDCD1": "immune_checkpoint", "CD274": "immune_checkpoint",
    "CTLA4": "immune_checkpoint", "LAG3": "immune_checkpoint",
    "VEGFA": "angiogenesis", "VEGFR2": "angiogenesis", "KDR": "angiogenesis",
    "FLT1": "angiogenesis", "FLT4": "angiogenesis",
    "EGFR": "growth_factor_kinase", "ERBB2": "growth_factor_kinase",
    "MET": "growth_factor_kinase", "ALK": "growth_factor_kinase",
    "RET": "growth_factor_kinase", "ABL1": "growth_factor_kinase",
    "BTK": "growth_factor_kinase", "JAK1": "growth_factor_kinase",
    "JAK2": "growth_factor_kinase", "FGFR1": "growth_factor_kinase",
    "FGFR2": "growth_factor_kinase", "FGFR3": "growth_factor_kinase",
    "KIT": "growth_factor_kinase", "PDGFRA": "growth_factor_kinase",
    "BRAF": "ras_mapk_pathway", "KRAS": "ras_mapk_pathway",
    "MAP2K1": "ras_mapk_pathway", "NRAS": "ras_mapk_pathway",
    "CDK4": "cell_cycle", "CDK6": "cell_cycle",
    "MTOR": "cell_cycle",
    "PARP1": "dna_damage_repair",
    "PSMB5": "proteostasis_immunomodulatory", "CRBN": "proteostasis_immunomodulatory",
    "AR": "hormonal", "ESR1": "hormonal",
    "CYP17A1": "hormonal", "CYP19A1": "hormonal",
    "MS4A1": "bcell_lymphoma_target", "TNFRSF8": "bcell_lymphoma_target",
    "BCL2": "bcell_lymphoma_target", "BCL6": "bcell_lymphoma_target",
    "DHFR": "cytotoxic_chemo", "RRM1": "cytotoxic_chemo",
    "TOP1": "cytotoxic_chemo", "TOP2A": "cytotoxic_chemo",
    "TUBB": "cytotoxic_chemo", "TYMS": "cytotoxic_chemo", "DNA": "cytotoxic_chemo",
}


def mechanism_family(target: str, modality: str = "") -> str:
    """Family for a target, falling back to a modality bucket for unlisted ones."""
    if target in TARGET_FAMILY:
        return TARGET_FAMILY[target]
    if target in ("UNKNOWN", "", None):
        return "unmapped"
    m = (modality or "").lower()
    if m in ("antibody", "bispecific", "adc"):
        return "other_biologic"
    if m == "cell_therapy":
        return "cell_therapy"
    return "other_small_molecule"


# ── Beta-Binomial empirical Bayes ──────────────────────────────────────────

def _fit_beta_prior_moments(rates: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """
    Method-of-moments Beta(alpha, beta) fit to a set of observed group rates.
    Standard empirical-Bayes step (this is the same approach used for e.g.
    shrinking small-sample batting averages toward the league mean).
    Falls back to a weak, uninformative-ish prior if variance can't be
    estimated (e.g. only one group, or all rates identical).
    """
    if len(rates) < 2 or weights.sum() <= 0:
        return 1.0, 1.0
    mean = np.average(rates, weights=weights)
    var = np.average((rates - mean) ** 2, weights=weights)
    mean = min(max(mean, 1e-3), 1 - 1e-3)
    if var <= 0 or var >= mean * (1 - mean):
        # degenerate variance — weak prior centred on the observed mean,
        # equivalent to ~2 pseudo-observations
        return mean * 2, (1 - mean) * 2
    common = mean * (1 - mean) / var - 1
    alpha = max(mean * common, 1e-3)
    beta = max((1 - mean) * common, 1e-3)
    return alpha, beta


def fit_hierarchical_rates(df: pd.DataFrame, label_col: str = "label_permissive",
                           ci: float = 0.90) -> pd.DataFrame:
    """
    Three-level Beta-Binomial shrinkage: pair <- mechanism family <- global.

    Returns one row per (target, disease) pair with:
      n, k                 raw trial count and positive count for this pair
      raw_rate             k/n (unshrunk — unstable for small n)
      posterior_mean       shrunk estimate — the number to actually use
      ci_lo, ci_hi         credible interval at the given level (default 90%)
      family               mechanism family used for pooling
      own_data_weight      0-1, how much the posterior relies on this pair's
                           own data vs. its family's average (n / (n + alpha+beta))
    """
    det = df[df[label_col] != -1].copy()
    det["_family"] = [mechanism_family(t, m) for t, m in
                      zip(det["primary_target"], det.get("modality", ""))]

    # Level 1: global prior across mechanism families
    fam_stats = det.groupby("_family")[label_col].agg(["mean", "count"])
    global_alpha, global_beta = _fit_beta_prior_moments(
        fam_stats["mean"].values, fam_stats["count"].values)

    # Level 2: per-family prior, itself shrunk toward the global prior using
    # a pseudo-count equal to the global prior's implied sample size
    # (alpha+beta) — a family with few pairs leans on the global rate; a
    # family with many leans on its own.
    global_pseudo_n = global_alpha + global_beta
    global_mean = global_alpha / global_pseudo_n

    family_priors = {}
    for fam, sub in det.groupby("_family"):
        pair_stats = sub.groupby(["primary_target", "disease"])[label_col].agg(
            ["mean", "count"])
        if len(pair_stats) >= 2:
            f_alpha, f_beta = _fit_beta_prior_moments(
                pair_stats["mean"].values, pair_stats["count"].values)
        else:
            f_alpha, f_beta = global_alpha, global_beta
        # blend the family's own prior with the global prior, weighted by how
        # much data the family itself has
        fam_n = sub[label_col].notna().sum()
        w = fam_n / (fam_n + global_pseudo_n)
        alpha = w * f_alpha + (1 - w) * global_alpha
        beta = w * f_beta + (1 - w) * global_beta
        family_priors[fam] = (alpha, beta)

    # Level 3: pair-level posterior = family prior + this pair's own k/n
    rows = []
    for (target, disease), sub in det.groupby(["primary_target", "disease"]):
        fam = sub["_family"].iloc[0]
        alpha, beta = family_priors.get(fam, (global_alpha, global_beta))
        n = len(sub)
        k = int(sub[label_col].sum())
        post_alpha, post_beta = alpha + k, beta + (n - k)
        post_mean = post_alpha / (post_alpha + post_beta)
        lo, hi = stats.beta.ppf([(1 - ci) / 2, 1 - (1 - ci) / 2], post_alpha, post_beta)
        rows.append({
            "primary_target": target, "disease": disease, "family": fam,
            "n": n, "k": k, "raw_rate": k / n if n else np.nan,
            "posterior_mean": post_mean, "ci_lo": float(lo), "ci_hi": float(hi),
            "prior_alpha": alpha, "prior_beta": beta,
            "own_data_weight": n / (n + alpha + beta),
        })

    result = pd.DataFrame(rows).sort_values("n", ascending=False)
    return result


def lookup_or_backoff(rates: pd.DataFrame, target: str, disease: str,
                      modality: str = "") -> dict:
    """
    Get the hierarchical rate for a specific pair. If the pair was never seen
    at all (a genuinely novel target-disease combination — the case this
    model matters most for), back off to the family-level prior mean, with
    own_data_weight=0 and the credible interval widened to reflect that
    nothing pair-specific is known.
    """
    hit = rates[(rates.primary_target == target) & (rates.disease == disease)]
    if len(hit):
        return hit.iloc[0].to_dict()
    fam = mechanism_family(target, modality)
    fam_rows = rates[rates.family == fam]
    if len(fam_rows):
        alpha, beta = fam_rows.iloc[0][["prior_alpha", "prior_beta"]]
    else:
        alpha, beta = 1.0, 1.0
    mean = alpha / (alpha + beta)
    lo, hi = stats.beta.ppf([0.05, 0.95], alpha, beta)
    return {
        "primary_target": target, "disease": disease, "family": fam,
        "n": 0, "k": 0, "raw_rate": np.nan,
        "posterior_mean": mean, "ci_lo": float(lo), "ci_hi": float(hi),
        "prior_alpha": alpha, "prior_beta": beta, "own_data_weight": 0.0,
    }


if __name__ == "__main__":
    from config import DATA_DIR, REPORT_DIR
    df = pd.read_csv(DATA_DIR / "classification_matrix.csv")
    df = df.loc[:, ~df.columns.duplicated()]
    rates = fit_hierarchical_rates(df)
    rates.to_csv(REPORT_DIR / "hierarchical_rates.csv", index=False)

    print(f"Fit hierarchical rates for {len(rates)} (target, disease) pairs "
          f"across {rates.family.nunique()} mechanism families\n")
    print(f"{'target':<10} {'disease':<22} {'family':<26} {'n':>3} {'raw':>6} "
          f"{'shrunk':>7} {'90% CI':<16} {'own_wt':>7}")
    for _, r in rates.sort_values("n").head(15).iterrows():
        raw = f"{r.raw_rate:.2f}" if r.n else "n/a"
        print(f"{r.primary_target:<10} {r.disease:<22} {r.family:<26} {r.n:>3} "
              f"{raw:>6} {r.posterior_mean:>7.3f} "
              f"[{r.ci_lo:.2f},{r.ci_hi:.2f}]      {r.own_data_weight:>7.2f}")
    print("\n  (smallest-n pairs shown — this is where shrinkage matters most:")
    print("   raw_rate is a coin-flip estimate from 1-3 trials; posterior_mean")
    print("   pulls it toward the mechanism family's rate, and own_data_weight")
    print("   close to 0 honestly says 'trust the family average here, not this pair'.)")
