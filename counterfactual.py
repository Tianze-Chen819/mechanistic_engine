"""
counterfactual.py — Counterfactual analysis for trial optimization.

v6: Updated to work with new pair-level feature structure.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import REPORT_DIR, FIG_DIR, log

SCORE_FEASIBILITY = {
    "TDS": "MEDIUM - combination therapy or target validation",
    "MCS": "MEDIUM - modality switch or formulation change",
    "EMS": "HIGH - invest in translational/mechanistic studies",
    "BFS": "HIGH - biomarker enrichment is a clinical design choice",
    "CDS": "LOW - requires fundamental biology change",
    "TWS": "MEDIUM - dose optimization or patient selection",
}

COMPOSITE_TO_RAW = {
    "TDS": ["tractability", "binding_evidence", "potency_proxy", "ot_known_drug", "depmap_essential"],
    "MCS": ["mechanistic_maturity", "pathway_evidence", "pubmed_pair_count", "ot_literature"],
    "EMS": ["human_study_fraction", "translational_study_fraction", "pair_pub_acceleration"],
    "BFS": ["alteration_frequency", "biomarker_directness", "lineage_specificity"],
    "CDS": ["cosmic_census_member", "somatic_evidence", "ot_genetic_association"],
    "TWS": ["tumor_expression", "tumor_specificity", "normal_tissue_burden"],
}

def _perturb_composite(X_row: pd.Series, score_name: str, delta: float = 0.1) -> pd.Series:
    """Perturb composite score by boosting its constituent raw features."""
    x = X_row.copy()
    raw_feats = COMPOSITE_TO_RAW.get(score_name, [])
    for feat in raw_feats:
        if feat in x.index:
            current = float(x[feat]) if x[feat] != -1 else 0.0
            x[feat] = min(current + delta, 1.0)
    if score_name in x.index:
        x[score_name] = min(float(x[score_name]) + delta, 1.0)
    return x

def run_counterfactual(best_models: dict, data: dict, n_trials: int = 50):
    """Run counterfactual analysis on test trials."""
    best = best_models.get("label_balanced")
    if not best or not best.get("model"):
        log.warning("No best model for counterfactual analysis")
        return

    model = best["model"]
    fs_name = best["features"]
    X_te, y_te = data["feature_sets"][fs_name][1], data["test_labels"]["label_balanced"]
    test_df = data["test_df"]

    # Select a sample
    sample_idx = X_te.index[:min(n_trials, len(X_te))]
    X_sample = X_te.loc[sample_idx]
    test_sample = test_df.loc[sample_idx]

    baseline_probs = model.predict_proba(X_sample)[:, 1]
    score_names = ["TDS", "MCS", "EMS", "BFS", "CDS", "TWS"]

    cf_records = []
    global_deltas = {s: [] for s in score_names}

    # Detailed output for first 3 trials
    for i, (idx, trial_row) in enumerate(test_sample.iterrows()):
        baseline_p = baseline_probs[list(sample_idx).index(idx)]
        drug = trial_row.get("canonical_drug", "unknown")
        disease = trial_row.get("disease", "unknown")
        nct = trial_row.get("nct_id", "")

        if i < 3:
            print(f"\n  --- {nct} | {drug} | {disease} ---")
            print(f"  {'score':<6}  {'baseline_p':<12}  {'counterfactual_p':<18}  {'delta_p':<10}  {'feasibility'}")

        for score_name in score_names:
            X_perturbed = _perturb_composite(X_sample.loc[idx], score_name)
            cf_prob = model.predict_proba(X_perturbed.values.reshape(1, -1))[0, 1]
            delta = cf_prob - baseline_p
            global_deltas[score_name].append(delta)

            if i < 3:
                feasibility = SCORE_FEASIBILITY.get(score_name, "UNKNOWN")
                print(f"  {score_name:<6}  {baseline_p:<12.4f}  {cf_prob:<18.4f}  {delta:<10.4f}  {feasibility}")

            cf_records.append({
                "nct_id": nct, "canonical_drug": drug, "disease": disease,
                "score": score_name, "baseline_p": round(baseline_p, 4),
                "counterfactual_p": round(cf_prob, 4), "delta_p": round(delta, 4),
                "feasibility": SCORE_FEASIBILITY.get(score_name, ""),
            })

    # Global summary
    global_summary = pd.DataFrame({
        "score": list(global_deltas.keys()),
        "mean_delta_p": [np.mean(v) for v in global_deltas.values()],
        "median_delta_p": [np.median(v) for v in global_deltas.values()],
        "std_delta_p": [np.std(v) for v in global_deltas.values()],
        "max_delta_p": [np.max(v) for v in global_deltas.values()],
        "pct_positive": [np.mean(np.array(v) > 0) * 100 for v in global_deltas.values()],
    }).set_index("score").sort_values("mean_delta_p", ascending=False)

    print(f"\n  === GLOBAL COUNTERFACTUAL ANALYSIS ===")
    print(global_summary.round(4).to_string())

    most_impactful = global_summary.index[0]
    print(f"\n  Most impactful lever: {most_impactful}")
    print(f"    -> {SCORE_FEASIBILITY.get(most_impactful, '')}")

    # Save
    pd.DataFrame(cf_records).to_csv(REPORT_DIR / "counterfactual_results.csv", index=False)
    global_summary.reset_index().to_csv(REPORT_DIR / "counterfactual_global.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1D9E75" if v > 0 else "#D85A30" for v in global_summary["mean_delta_p"]]
    ax.barh(global_summary.index, global_summary["mean_delta_p"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean Δ predicted probability")
    ax.set_title("Counterfactual impact by optimization lever")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "counterfactual_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Counterfactual plots saved")
    log.info(f"Counterfactual analysis complete. Most impactful: {most_impactful}")