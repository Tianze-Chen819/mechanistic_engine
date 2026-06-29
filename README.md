# Mechanistic Phase 2 Biology Engine

A standalone analytical engine that predicts Phase 2 clinical trial success from biology/mechanism alone.

## What it does

Estimates **P(positive Phase 2 proof-of-concept | biology/mechanism only)** by computing mechanistic features from public databases and training classification models.

## Data Sources

| Source | What it provides | Method |
|--------|-----------------|--------|
| ClinicalTrials.gov | Trial corpus (5,000+ Phase 2 oncology trials) | REST API |
| Open Targets | Target-disease association scores | GraphQL API |
| PubMed | Publication counts and evidence maturity | NCBI E-utilities |
| Curated dictionary | Drug-target-modality mappings for 100+ drugs | Local lookup |

## Installation

```bash
pip install -r requirements.txt
```

## Running the Pipeline

```bash
python run_pipeline.py
```

The pipeline takes ~15-20 minutes (mostly API queries, cached after first run).

To skip API calls and use curated data only (faster, ~2 minutes):
- Edit `config.py` and set `USE_REAL_APIS = False`

## Optional Deep Learning Branch

The default pipeline trains the existing RandomForest/LightGBM baselines. An
optional complementary branch adds learned drug/target/disease embeddings with a
small MLP and TrialTransformer. It does not change `labels.py` or replace the
tree models.

```bash
pip install -r requirements-deep.txt
RUN_DEEP_EXPERIMENTS=1 python run_pipeline.py
```

After `run_pipeline.py` has produced the feature matrix, the deep branch can
also be run standalone:

```bash
python run_deep_experiments.py
```

More details are in `docs/deep_learning_branch.md`.

## Output Files

| File | Description |
|------|-------------|
| `data/classification_matrix.csv` | Main file: 3000+ trials × 72 columns (features + labels) |
| `data/classification_matrix.parquet` | Same in Parquet format |
| `data/data_dictionary.csv` | Description of every column |
| `reports/model_comparison.csv` | All model metrics (ROC AUC, PR AUC, F1, Brier, etc.) |
| `reports/cohort_report.csv` | Cohort-level summary statistics |
| `reports/asset_report.csv` | Per-trial predictions with score breakdowns |
| `reports/error_analysis.csv` | False positive / false negative analysis |
| `reports/cross_validation.csv` | 5-fold cross-validation results |
| `reports/sensitivity_analysis.csv` | Sensitivity analysis across subsets |
| `reports/counterfactual_results.csv` | Counterfactual perturbation results |
| `figures/*.png` | ROC curves, calibration, SHAP, confusion matrices, etc. |

## Project Structure

```
config.py              - Settings, paths, feature lists
drug_target_db.py      - Curated drug-target-modality dictionary
clients.py             - API clients (ClinicalTrials.gov, Open Targets, PubMed)
normalize.py           - Drug/disease/biomarker normalization
labels.py              - Label construction (strict/balanced/permissive)
features.py            - Raw feature computation + composite scores
modeling.py            - Model training, evaluation, calibration
counterfactual.py      - Counterfactual analysis module
reporting.py           - Reports, sensitivity analysis, cross-validation
run_pipeline.py        - Main pipeline runner
run_deep_experiments.py - Optional standalone deep-learning runner
requirements.txt       - Python dependencies
requirements-deep.txt  - Optional PyTorch dependency for deep models
README.md              - This file
```

## Models Trained

- Logistic Regression (with isotonic calibration)
- LightGBM
- Random Forest

Each trained across 3 feature sets (raw, composite, hybrid) × 3 label definitions (strict, balanced, permissive) = 27+ model configurations.

## Label Definitions

- **Strict (~9%)**: Only clearly positive results
- **Balanced (~33%)**: Positive results + strong biology completions
- **Permissive (~74%)**: Any completion or continuation signal

## Key Results

- Best model: AUC ~0.87 on balanced labels
- Top features: target-disease association, normal tissue burden, dependency mean
- Most actionable counterfactual lever: Therapeutic Window Score

## Notes

- No API keys required (all public APIs)
- All API results are cached locally in `mechanistic_engine_output/cache/`
- Second runs are fast because of caching
- This is a biology-only model; it intentionally ignores protocol design and site execution
