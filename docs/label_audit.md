# Label audit and evaluation of the original data

Written for: whoever picks up this engine next and needs to know what its
numbers mean before relying on them.

Reproduce with `python audit_labels.py` and `python evaluate_original.py`.
Machine-readable results land in `reports/label_audit.csv` and
`reports/evaluation_original.csv`.

---

## Summary

The engine is built to test whether biology alone predicts Phase 2 success. As
shipped it cannot answer that question, because the thing it is trained to
predict is not a Phase 2 outcome. All eight label checks fail.

Separately, two of the four pair-level data sources were returning nothing:
cBioPortal supplied real data for **0 of 638** trials, and PubMed's counts were
both saturated and temporally leaky. Both are now repaired.

---

## Part 1 — What the label actually encodes

### Every positive comes from text written before the trial started

`labels.py` scores a trial positive when a phrase from `POSITIVE_TEXT_TRIGGERS`
appears in `why_stopped + brief_summary + primary_outcome`. Only `why_stopped`
is written after the fact. The other two are authored at registration.

Across all 123 positives, **123 (100%) fired only on `brief_summary` or
`primary_outcome`. Not one fired on `why_stopped`.**

The two dominant triggers show why this is fatal:

| Trigger | Field | Firings | What it is really matching |
|---|---|---|---|
| `approved` | brief_summary | 53 | background prose about *other* drugs |
| `complete response` | primary_outcome | 52 | the RECIST endpoint *definition* |
| `partial response` | primary_outcome | 44 | the RECIST endpoint *definition* |
| `objective response` | brief_summary + outcome | 52 | the endpoint being measured |
| `clinical benefit` | brief_summary | 20 | the endpoint being measured |

All 276 trigger firings across all 123 positives land in `brief_summary` (149)
or `primary_outcome` (127). `why_stopped` contributes zero.

Real examples from the corpus, all labelled **positive**:

- `NCT01853748` — "the FDA has **not approved** this drug for use in patients
  undergoing adjuvant treatment for HER2+ breast cancer". The matcher has no
  negation handling, so an explicit statement of non-approval scores positive.
- `NCT00183885` — "Mitomycin-C is a drug that has been approved by the FDA to
  treat cancer of the stomach and pancreas." Background about a 1970s
  chemotherapy agent, in a trial about liver-directed therapy.
- `NCT00821327` — "Complete Response (CR): Disappearance of all target lesions.
  Partial Response (PR): At least a 30% decrease…". This is boilerplate present
  in essentially every solid-tumour protocol ever registered.

The label is measuring how a protocol was **written**, not how the trial turned
out. A confirming test: raw character count of the pre-trial text predicts the
label at **AUC 0.638** — higher than any model reaches on the temporal split.
Verbose protocols define their endpoints at length; long endpoint definitions
contain more RECIST vocabulary; more vocabulary means more trigger hits.

### One heuristic produces three quarters of the labels

| Provenance rule | n | Label | Share |
|---|---|---|---|
| `completed+results+no_positive_signal` | 479 | 0 | **75%** |
| `completed+positive_text+results` | 94 | 1 | 15% |
| `completed+positive_text+approved` | 29 | 1 | 5% |
| `terminated+efficacy` | 23 | 0 | 4% |
| `terminated+negative_text` | 9 | 0 | 1% |
| `completed+negative_text` | 3 | 0 | <1% |
| `withdrawn+negative` | 1 | 0 | <1% |

93% of the negative class is a single rule that means "results were posted and
no marketing phrase appeared." Posting results is a legal obligation under
FDAAA 801, not an outcome. Only **5.2%** of labels rest on a recorded post-hoc
event of any kind.

### The three label definitions are one label

`labels.py:244` sets `balanced = strict`, and the `permissive` branch at line
249 is guarded by `if permissive == -1`, so it can only touch rows that are
already indeterminate — and every branch inside it either assigns `-1` or
requires `approved_here`. Result: `label_strict`, `label_balanced` and
`label_permissive` are **100% identical** across all 638 rows.

`reports/model_comparison.csv` therefore has 45 rows containing 9 distinct
results: the label axis is a threefold duplicate, and `hybrid` is byte-identical
to `composite`.

### The trigger matcher has two mechanical bugs

**Negation blindness.** `_text_signal` is a bare substring search. All five of
these score `positive`:

```
"The FDA has not approved this drug for this indication."
"This study was approved by the institutional review board."
"Primary endpoint: proportion with complete response or partial response per RECIST 1.1."
"If successful, this will support a phase 3 study."
```

**Bare `"efficacy"` in `TERMINATED_EFFICACY_REASONS`.** The list contains the
substring `"efficacy"` alongside `"lack of efficacy"`, so it matches successful
terminations too. Three of these four score as *negative*:

```
"Drug demonstrated efficacy at interim analysis; moved to Phase 3"   -> negative
"Efficacy established, transitioned to registration trial"            -> negative
"Study closed early; primary efficacy endpoint met"                   -> negative
```

Since `terminated+efficacy` is the only rule in the whole system backed by hard
post-hoc evidence, corrupting it removes the last real outcome signal.

### Two smaller issues

- `assign_labels(trial, known_approvals)` never uses `known_approvals`.
- `ACTIVE_NOT_RECRUITING` is listed in `LABELABLE_STATUSES` but falls through to
  `else: strict = -1`, so it is never labelled.

---

## Part 2 — Sample size and split integrity

638 rows is the headline, but they collapse onto **200 distinct (target,
disease) pairs** across **34 targets**. The biology features are a pure function
of that pair, so the effective sample size is 200, not 638. 23% of rows have a
chemotherapy pseudo-target (`DNA`, `TUBB`, `UNKNOWN`) for which target-level
biology features are meaningless.

The temporal split at 2015 is not grouped:

| Overlap | Share of test rows |
|---|---|
| shares a (drug, disease) pair with train | 43% |
| drug appears in train | 86% |
| target appears in train | 98% |

Those 43% have near-identical feature vectors to training rows, so the model can
memorise instead of generalise.

---

## Part 3 — Repairs to cBioPortal and PubMed

### cBioPortal: was returning nothing at all

`has_real_cbio_data` was `False` for **all 638 rows**, and
`alteration_frequency`, `co_alteration_burden` and `genomic_complexity` were
entirely NaN. Two dead endpoints:

1. `GET /studies?cancerTypeId=breast` — cBioPortal does not filter studies by
   cancer type on this route. It silently ignores the parameter and returns
   `[]`, so the function bailed on its first call every time.
2. `GET /genes/{symbol}/mutations` — 404, the route does not exist.

Even had they worked, `lineage_specificity`, `co_alteration_burden` and
`genomic_complexity` were assigned the constants 0.6, 0.4 and 0.5.

The repaired client uses the verified call sequence (`/genes/{symbol}` for the
Entrez id, a locally-filtered study list, `/mutations/fetch` plus
`/discrete-copy-number/fetch` for the numerator, `_sequenced`/`_cna` sample
lists for the denominator) and computes all four values for real. Diseases map
to the matching TCGA PanCancer Atlas cohort, with an OncoTree subtree search for
the haematological malignancies TCGA barely covers.

Validation against known biology:

| Pair | Computed | Published |
|---|---|---|
| ERBB2 / breast | 13.7% | ~13–15% HER2-amplified |
| TP53 / breast | 33.1% | ~33% |
| EGFR / NSCLC | 14.8% | ~14% in Western cohorts |
| BRAF / melanoma | 53.4% | ~50% |
| KRAS / pancreatic | 63.9% | ~70–90% (TCGA PAAD runs low on purity) |

`lineage_specificity` now behaves correctly too: BRAF/melanoma 0.93 (highly
lineage-restricted), TP53/breast 0.48 (pan-cancer, no preference), EGFR/breast
0.27 (not a breast gene).

### PubMed: saturated, leaky, and missing most of its literature

Three defects:

1. **Saturation.** `pubmed_pair_count = min(count/1000, 1.0)` pinned every
   well-studied pair to exactly 1.0 — EGFR/NSCLC has ~20,000 papers.
   `clinical_trial_pub_count = min(count/100, 1.0)` collapsed 638 trials onto 41
   distinct values. Both are now log-scaled.

2. **Temporal leakage.** Counts were taken as of today for trials that started
   as early as 1994. For EGFR/NSCLC, ~80% of the corpus postdates 2014 — that
   literature includes papers reporting the outcome of the very trial being
   predicted. Every query is now bounded at the trial's start year, and
   `pair_pub_acceleration`'s hardcoded `2021:2024` window (which for a 2010
   trial measured publications a decade after the fact) is now a
   four-year window ending at trial start.

3. **Missing aliases.** The clinical literature uses protein names, not HGNC
   symbols. `MS4A1[Title/Abstract] AND lymphoma` returns 30 papers; adding the
   `CD20` alias returns 8,105. Multi-word disease names were also passed
   unquoted, so `breast cancer[Title/Abstract]` parsed as two tokens. Diseases
   now expand to a MeSH heading plus quoted synonyms.

Making the features year-aware changed the pair key throughout: `enrich_all`
takes `(target, disease, start_year)` triples, and `features.py` looks up the
triple with a fallback to the old two-tuple. Open Targets and cBioPortal results
are memoised per `(target, disease)` since they do not depend on the year.

### One bug affecting every client

`cached_get` returned `None` on any `HTTPError`, including 429. NCBI rate-limits
aggressively, so throttled requests silently became zero-valued features rather
than being retried. It now retries 429 and 5xx with backoff, honouring
`Retry-After`. Set `NCBI_API_KEY` to raise the rate limit from 3/s to 10/s.

---

## Part 4 — Evaluation of the original data

`evaluate_original.py` scores every model under four protocols against three
null baselines, with 1,000-sample bootstrap CIs and permutation tests.

Null baselines first, because they set the bar:

| Baseline | AUC |
|---|---|
| Predict base rate | 0.500 |
| Count of non-missing biology features (data availability, zero mechanism) | 0.508 |
| Missingness indicators only | 0.513 |
| **Pre-trial text character count** | **0.638** |

Best model per protocol:

| Protocol | Best AUC | 95% CI | Perm p | CI clears null |
|---|---|---|---|---|
| Temporal split (pipeline default) | 0.538 | 0.433–0.636 | 0.24 | no |
| Temporal, unseen pairs only (n=93) | 0.606 | 0.469–0.739 | — | no |
| Grouped 5-fold CV on (target,disease) | 0.612 | 0.550–0.679 | 0.002 | yes |
| Stratified CV (optimistic bound) | 0.640 | 0.588–0.691 | — | yes |

Three things follow.

**The pipeline's own protocol has no detectable signal.** On the temporal split
the best model reaches 0.538 with a CI spanning 0.43–0.64 and a permutation
p-value of 0.24. With 198 test rows and 38 positives, nothing in that range is
distinguishable from chance. This is also why the discarded README claim of
"AUC ~0.87" was never reachable — `model_comparison.csv` itself tops out at
0.633.

**Grouped CV does find something small and real** (0.612, p=0.002), because it
uses all 638 rows instead of 198 and never lets a fold's biology appear in its
own training set. But it is barely above the 0.638 that pre-trial text length
achieves on its own, which is the reading to be careful about.

**The surviving signal is genomic, not bibliometric.** Under grouped CV:

| Feature set | Best AUC | Perm p |
|---|---|---|
| genomic only (COSMIC / DepMap / GWAS / somatic) | 0.612 | 0.002 |
| all raw features | 0.576 | 0.007 |
| raw minus literature | 0.581 | 0.002 |
| literature only | 0.535 | 0.15 (n.s.) |
| composite scores | 0.534 | 0.10 (n.s.) |

Literature features do not reach significance, and removing them does not hurt.
The seven hand-built composite scores are *worse* than the raw features they
summarise. This is the one encouraging result in the audit — and it is also the
part the cBioPortal repair strengthens directly, since it adds four real
genomic features to the exact set that already carries the signal.

---

## What to do next

In priority order:

1. **Rebuild the label from results, not prose.** ClinicalTrials.gov exposes a
   structured results section (`resultsSection.outcomeMeasures`) with actual
   measured values. That is the outcome. Failing that, link trials to their
   publication via `clients_v8_additions.fetch_pubmed_by_nct` and read the
   conclusion. Either is a real label; substring-matching a protocol is not.
2. **Restrict text triggers to `why_stopped`.** If prose matching must stay, it
   should only read the one field written after the trial ended, and it needs
   negation handling.
3. **Fix `TERMINATED_EFFICACY_REASONS`** — drop the bare `"efficacy"` entry.
4. **Group the split** on `(target, disease)`, not just on year.
5. **Re-run enrichment** so the repaired cBioPortal and date-bounded PubMed
   features are in the matrix, then re-run `evaluate_original.py`. The cache is
   keyed by new tags, so the pair-level queries will re-fetch (~20 min with an
   `NCBI_API_KEY`, longer without).
6. **Report intervals, not point estimates.** At n=200 effective, a bare AUC is
   not a result.

Until step 1 is done, every performance number in this repository — including
the honest ones above — describes how well biology predicts the way a protocol
was worded.
