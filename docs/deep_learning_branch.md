# Optional Deep Learning Branch

This branch is a complement to the existing RandomForest and LightGBM baselines.
It does not replace the tree models and does not change label construction.

## What It Adds

The first implementation trains two small local models:

- `DeepMLP`: learned categorical embeddings plus the current biology feature vector.
- `TrialTransformer`: a small transformer encoder over trial entity tokens plus a biology token.

The learned entity embeddings cover:

- `canonical_drug`
- `primary_target`
- `disease`
- `modality`
- `moa`

All vocabularies are fit only on training rows. Unseen validation or test
entities map to `<UNK>`.

## Evaluation

The branch uses the same temporal split as the baseline pipeline:

- train: trials with `start_year <= TEST_YEAR_CUTOFF`
- test: trials with `start_year > TEST_YEAR_CUTOFF`

Model selection uses a chronological validation split inside the training set.
The post-2015 test set is used only once for final evaluation.

## Running

Install the optional dependency:

```bash
.venv/bin/pip install -r requirements-deep.txt
```

Run with the full pipeline:

```bash
RUN_DEEP_EXPERIMENTS=1 MPLCONFIGDIR=/private/tmp/mpl-cache .venv/bin/python run_pipeline.py
```

Run standalone after the feature matrix exists:

```bash
MPLCONFIGDIR=/private/tmp/mpl-cache .venv/bin/python run_deep_experiments.py
```

Smoke test:

```bash
MPLCONFIGDIR=/private/tmp/mpl-cache .venv/bin/python run_deep_experiments.py --smoke
```

If the standalone script cannot find the required feature matrix, run:

```bash
.venv/bin/python run_pipeline.py
```

## Outputs

Files are written to `mechanistic_engine_output/reports/`:

- `deep_model_comparison.csv`
- `deep_predictions.csv`
- `ensemble_comparison.csv`

## Future Pretrained Embeddings

The default implementation does not download or require ChemBERTa, ESM, RDKit,
`transformers`, or any large pretrained model. Future local pretrained vectors
can be connected through `ExternalEmbeddingProvider` in `embedding_features.py`.
