"""
audit_labels.py — Adversarial audit of the outcome labels.

The engine's headline claim is that biology predicts Phase 2 success. That claim
is only as good as the labels. This script interrogates them directly and writes
`reports/label_audit.csv` plus a console summary.

Run:
    python audit_labels.py                  # audit the shipped matrix
    python audit_labels.py --sample 200     # re-fetch more trial text from CT.gov

Checks performed
  1. Label agreement      — are strict/balanced/permissive actually different?
  2. Provenance mix       — which rule produced each label, and how concentrated?
  3. Trigger provenance   — for positives, WHICH phrase fired, and in WHICH field.
                            A trigger that fires on brief_summary or
                            primary_outcome fired on text written before the
                            trial started, so it cannot be outcome evidence.
  4. Verbosity confound   — how well does raw text length alone predict the label?
  5. Trigger unit tests   — negation blindness, bare-substring over-matching.
  6. Split contamination  — do (drug, disease) pairs straddle the temporal split?
  7. Effective sample size — distinct biology profiles vs row count.
"""

import argparse
import collections
import json
import random
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

from config import (
    DATA_DIR, REPORT_DIR, CACHE_DIR, TEST_YEAR_CUTOFF,
    POSITIVE_TEXT_TRIGGERS, NEGATIVE_TEXT_TRIGGERS, TERMINATED_EFFICACY_REASONS,
)

CT_API = "https://clinicaltrials.gov/api/v2/studies"
TEXT_CACHE = CACHE_DIR / "audit_trial_text.json"

# Fields the labeller reads, split by when they were written. Anything in
# PRE_TRIAL_FIELDS is authored at registration, before a single patient enrols.
PRE_TRIAL_FIELDS = ("brief_summary", "primary_outcome")
POST_HOC_FIELDS = ("why_stopped",)


def _rule(name, verdict, detail):
    return {"check": name, "verdict": verdict, "detail": detail}


# ── 1-2. Label agreement and provenance ───────────────────────────────────────

def check_label_agreement(df, findings):
    print("\n" + "=" * 78)
    print("1. LABEL AGREEMENT — are the three label definitions distinct?")
    print("=" * 78)
    cols = ["label_strict", "label_balanced", "label_permissive"]
    for c in cols:
        vc = df[c].value_counts().to_dict()
        det = df[df[c] != -1]
        rate = det[c].mean() if len(det) else float("nan")
        print(f"  {c:<20} pos={vc.get(1,0):>4} neg={vc.get(0,0):>4} "
              f"indet={vc.get(-1,0):>4}  pos_rate={rate:.1%}")

    identical = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            same = (df[a] == df[b]).mean()
            print(f"  {a} vs {b}: {same:.1%} identical")
            if same == 1.0:
                identical.append((a, b))

    if identical:
        print("\n  ** All label definitions collapse to the same vector. **")
        print("     labels.py sets `balanced = strict`, and the permissive branch")
        print("     only fires for rows already determinate, so it adds nothing.")
        print("     Every 'label definition' axis in model_comparison.csv is a")
        print("     duplicate of the same run.")
        findings.append(_rule(
            "label_agreement", "FAIL",
            f"{len(identical)} label pairs are 100% identical; the 3 advertised "
            f"label definitions are 1 label evaluated 3 times"))
    else:
        findings.append(_rule("label_agreement", "PASS",
                              "label definitions are materially different"))


def check_provenance(df, findings):
    print("\n" + "=" * 78)
    print("2. PROVENANCE MIX — which rule assigns each label?")
    print("=" * 78)
    det = df[df.label_permissive != -1]
    tab = (det.groupby(["label_provenance", "label_permissive"]).size()
              .unstack(fill_value=0))
    tab["total"] = tab.sum(axis=1)
    tab["share"] = tab["total"] / len(det)
    print(tab.sort_values("total", ascending=False).to_string(
        float_format=lambda x: f"{x:.3f}"))

    top_rule, top_n = tab["total"].idxmax(), tab["total"].max()
    share = top_n / len(det)
    print(f"\n  Single most common rule: '{top_rule}' -> {top_n}/{len(det)} ({share:.0%})")
    if share > 0.4:
        print(f"  ** One heuristic supplies {share:.0%} of all labels. The model is")
        print(f"     largely learning to reproduce that one rule. **")
        findings.append(_rule(
            "provenance_concentration", "FAIL",
            f"'{top_rule}' produces {share:.0%} of determinate labels"))

    # How many labels rest on genuine post-hoc outcome evidence?
    hard = det.label_provenance.str.contains("terminated|withdrawn", regex=True)
    print(f"\n  Labels backed by a recorded post-hoc event (termination/withdrawal): "
          f"{hard.sum()}/{len(det)} ({hard.mean():.1%})")
    print(f"  Labels resting only on registration-time free text: "
          f"{(~hard).sum()}/{len(det)} ({(~hard).mean():.1%})")
    findings.append(_rule(
        "hard_outcome_evidence",
        "FAIL" if hard.mean() < 0.2 else "PASS",
        f"only {hard.mean():.1%} of labels have a recorded post-hoc outcome event"))


# ── 3. Trigger provenance (needs trial text) ──────────────────────────────────

def fetch_trial_text(nct_ids, force=False):
    cache = {}
    if TEXT_CACHE.exists() and not force:
        try:
            cache = json.loads(TEXT_CACHE.read_text())
        except Exception:
            cache = {}
    missing = [n for n in nct_ids if n not in cache]
    if missing:
        print(f"  fetching {len(missing)} trial records from ClinicalTrials.gov...")
        session = requests.Session()
        session.headers.update({"User-Agent": "MechanisticEngine-audit/1.0"})
        for i, nct in enumerate(missing, 1):
            try:
                r = session.get(f"{CT_API}/{nct}", timeout=30)
                if r.status_code != 200:
                    continue
                ps = r.json().get("protocolSection", {})
                outcomes = ps.get("outcomesModule", {}).get("primaryOutcomes", []) or []
                cache[nct] = {
                    "brief_summary": ps.get("descriptionModule", {}).get("briefSummary", "") or "",
                    "primary_outcome": " ".join(
                        f"{o.get('measure','')} {o.get('description','')}" for o in outcomes),
                    "why_stopped": ps.get("statusModule", {}).get("whyStopped", "") or "",
                }
            except Exception:
                continue
            if i % 25 == 0:
                print(f"    {i}/{len(missing)}")
            time.sleep(0.05)
        TEXT_CACHE.write_text(json.dumps(cache))
    return {n: cache[n] for n in nct_ids if n in cache}


def check_trigger_provenance(df, text, findings):
    print("\n" + "=" * 78)
    print("3. TRIGGER PROVENANCE — what evidence actually created each positive?")
    print("=" * 78)
    pos_ids = [n for n in df[df.label_permissive == 1].nct_id if n in text]
    if not pos_ids:
        print("  no text available; skipping")
        return

    by_field = collections.Counter()
    by_trigger = collections.Counter()
    pretrial_only = 0
    for nct in pos_ids:
        t = text[nct]
        fired_post, fired_pre = False, False
        for trig in POSITIVE_TEXT_TRIGGERS:
            for field in PRE_TRIAL_FIELDS + POST_HOC_FIELDS:
                if trig in (t.get(field, "") or "").lower():
                    by_field[field] += 1
                    by_trigger[(trig, field)] += 1
                    if field in POST_HOC_FIELDS:
                        fired_post = True
                    else:
                        fired_pre = True
        if fired_pre and not fired_post:
            pretrial_only += 1

    print(f"  positives examined: {len(pos_ids)}")
    print(f"\n  trigger firings by source field:")
    for field, n in by_field.most_common():
        when = "PRE-TRIAL (written at registration)" if field in PRE_TRIAL_FIELDS \
               else "post-hoc"
        print(f"    {field:<18} {n:>4}   {when}")

    print(f"\n  top (trigger, field) combinations:")
    for (trig, field), n in by_trigger.most_common(12):
        flag = "  <-- pre-trial text" if field in PRE_TRIAL_FIELDS else ""
        print(f"    {trig:<32} {field:<17} {n:>4}{flag}")

    frac = pretrial_only / len(pos_ids)
    print(f"\n  positives whose ONLY evidence is pre-trial text: "
          f"{pretrial_only}/{len(pos_ids)} ({frac:.0%})")
    if frac > 0.5:
        print("  ** These labels cannot be outcome labels. `primary_outcome` is the")
        print("     endpoint DEFINITION ('Objective Response Rate per RECIST 1.1'),")
        print("     which is boilerplate in essentially every oncology protocol, and")
        print("     `brief_summary` background prose mentions prior FDA approvals of")
        print("     other drugs. The label is measuring how a protocol was WRITTEN,")
        print("     not how the trial turned out. **")
        findings.append(_rule(
            "positive_label_validity", "FAIL",
            f"{frac:.0%} of positives derive solely from text written before the "
            f"trial started"))
    else:
        findings.append(_rule("positive_label_validity", "PASS",
                              "positives rest on post-hoc evidence"))


# ── 4. Verbosity confound ─────────────────────────────────────────────────────

def check_verbosity_confound(df, text, findings):
    print("\n" + "=" * 78)
    print("4. VERBOSITY CONFOUND — does text LENGTH alone predict the label?")
    print("=" * 78)
    from sklearn.metrics import roc_auc_score

    rows = []
    for _, r in df[df.label_permissive != -1].iterrows():
        t = text.get(r.nct_id)
        if not t:
            continue
        n = len(t.get("brief_summary", "")) + len(t.get("primary_outcome", ""))
        rows.append((r.label_permissive, n))
    if len(rows) < 40:
        print("  not enough text fetched; re-run with --sample 400")
        return

    y = np.array([a for a, _ in rows])
    n = np.array([b for _, b in rows])
    if len(set(y)) < 2:
        return
    auc = roc_auc_score(y, n)
    print(f"  n={len(rows)}  median chars: pos={np.median(n[y==1]):.0f} "
          f"neg={np.median(n[y==0]):.0f}")
    print(f"  AUC of raw character count alone: {auc:.3f}")
    if auc > 0.60:
        print(f"  ** A feature with no biology in it — how many characters the")
        print(f"     sponsor wrote — reaches AUC {auc:.2f}. Any model AUC at or")
        print(f"     below this is explained by protocol verbosity, not mechanism. **")
        findings.append(_rule("verbosity_confound", "FAIL",
                              f"text length alone gives AUC {auc:.3f}"))
    else:
        findings.append(_rule("verbosity_confound", "PASS",
                              f"text length gives AUC {auc:.3f}"))


# ── 5. Trigger logic unit tests ───────────────────────────────────────────────

def check_trigger_logic(findings):
    print("\n" + "=" * 78)
    print("5. TRIGGER LOGIC — does the matcher mean what it says?")
    print("=" * 78)
    from labels import _text_signal, _termination_is_efficacy

    cases = [
        # (text, what a clinician would say, what the code says)
        ("The FDA has not approved this drug for this indication.", "negative/neutral"),
        ("This study was approved by the institutional review board.", "neutral"),
        ("Mitomycin-C has been approved by the FDA to treat stomach cancer.", "neutral"),
        ("Primary endpoint: proportion with complete response or partial response "
         "per RECIST 1.1.", "neutral"),
        ("If successful, this will support a phase 3 study.", "neutral"),
    ]
    bad = 0
    print("  _text_signal() on text that carries no outcome information:")
    for txt, expected in cases:
        got = _text_signal(txt)
        wrong = got == "positive"
        bad += wrong
        print(f"    [{got:<8}] {'<-- WRONG' if wrong else '         '} {txt[:64]}")

    print("\n  _termination_is_efficacy() — TERMINATED_EFFICACY_REASONS contains the")
    print("  bare substring 'efficacy', so it matches successes too:")
    stops = [
        ("Terminated due to lack of efficacy", True),
        ("Drug demonstrated efficacy at interim analysis; moved to Phase 3", False),
        ("Efficacy established, transitioned to registration trial", False),
        ("Study closed early; primary efficacy endpoint met", False),
    ]
    bad_stop = 0
    for txt, should_be_negative in stops:
        got = _termination_is_efficacy(txt)
        wrong = got and not should_be_negative
        bad_stop += wrong
        print(f"    [negative={got!s:<5}] {'<-- WRONG' if wrong else '         '} {txt}")

    if bad or bad_stop:
        print(f"\n  ** {bad} of {len(cases)} neutral strings are scored positive; "
              f"{bad_stop} of {len(stops)} successful terminations are scored negative.")
        print("     The matcher is a bare substring search with no negation handling"
              "\n     and no scope restriction to results-bearing fields. **")
        findings.append(_rule(
            "trigger_logic", "FAIL",
            f"{bad} false-positive and {bad_stop} false-negative trigger cases; "
            f"no negation handling, 'efficacy' matches as a bare substring"))
    else:
        findings.append(_rule("trigger_logic", "PASS", "trigger matching is sound"))


# ── 6-7. Split contamination and effective sample size ────────────────────────

def check_split_and_dedup(df, findings):
    print("\n" + "=" * 78)
    print("6. SPLIT CONTAMINATION — does the temporal split isolate the test set?")
    print("=" * 78)
    det = df[df.label_permissive != -1].copy()
    tr = det[det.start_year < TEST_YEAR_CUTOFF]
    te = det[det.start_year >= TEST_YEAR_CUTOFF]
    print(f"  cutoff {TEST_YEAR_CUTOFF}: train n={len(tr)} ({tr.label_permissive.mean():.1%} pos)"
          f"  test n={len(te)} ({te.label_permissive.mean():.1%} pos)")

    shared_pairs = set(zip(tr.canonical_drug, tr.disease)) & set(zip(te.canonical_drug, te.disease))
    leak = te.apply(lambda r: (r.canonical_drug, r.disease) in shared_pairs, axis=1).mean()
    drug_leak = te.canonical_drug.isin(set(tr.canonical_drug)).mean()
    tgt_leak = te.primary_target.isin(set(tr.primary_target)).mean()
    print(f"  test rows sharing a (drug, disease) pair with train: {leak:.0%}")
    print(f"  test rows whose drug appears in train:               {drug_leak:.0%}")
    print(f"  test rows whose target appears in train:             {tgt_leak:.0%}")
    if leak > 0.25:
        print("  ** The split is temporal but not grouped. Because the biology")
        print("     features are functions of (target, disease) only, a test row")
        print("     that shares a pair with a training row has a near-identical")
        print("     feature vector — the model can memorise rather than generalise. **")
        findings.append(_rule("split_contamination", "FAIL",
                              f"{leak:.0%} of test rows share a (drug, disease) pair with train"))
    else:
        findings.append(_rule("split_contamination", "PASS", "split is clean"))

    print("\n" + "=" * 78)
    print("7. EFFECTIVE SAMPLE SIZE — how many distinct biology profiles?")
    print("=" * 78)
    n_pairs = det.groupby(["primary_target", "disease"]).ngroups
    n_tgt = det.primary_target.nunique()
    print(f"  rows: {len(det)}")
    print(f"  distinct (target, disease) pairs: {n_pairs}")
    print(f"  distinct targets: {n_tgt}")
    print(f"  rows per distinct pair: {len(det)/n_pairs:.1f}")
    print("\n  top targets by row count:")
    for t, n in det.primary_target.value_counts().head(8).items():
        print(f"    {t:<12} {n:>4}")
    pseudo = det.primary_target.isin(["DNA", "UNKNOWN", "TUBB"]).mean()
    print(f"\n  rows whose 'target' is a chemotherapy pseudo-target "
          f"(DNA/TUBB/UNKNOWN): {pseudo:.0%}")
    findings.append(_rule(
        "effective_sample_size",
        "FAIL" if n_pairs < len(det) / 2 else "PASS",
        f"{len(det)} rows collapse onto {n_pairs} distinct biology profiles "
        f"({n_tgt} targets); {pseudo:.0%} are chemo pseudo-targets"))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=250,
                    help="how many trials to pull text for (0 = all)")
    ap.add_argument("--refresh", action="store_true", help="ignore the text cache")
    args = ap.parse_args()

    path = DATA_DIR / "classification_matrix.csv"
    if not path.exists():
        sys.exit(f"missing {path} — run run_pipeline.py first")
    df = pd.read_csv(path)
    print(f"loaded {len(df)} labelled trials from {path}")

    findings = []
    check_label_agreement(df, findings)
    check_provenance(df, findings)

    det = df[df.label_permissive != -1]
    ids = det.nct_id.tolist()
    if args.sample and args.sample < len(ids):
        random.seed(0)
        pos = det[det.label_permissive == 1].nct_id.tolist()
        neg = det[det.label_permissive == 0].nct_id.tolist()
        half = args.sample // 2
        ids = (random.sample(pos, min(half, len(pos))) +
               random.sample(neg, min(args.sample - half, len(neg))))
    print(f"\n  resolving trial text for {len(ids)} trials...")
    text = fetch_trial_text(ids, force=args.refresh)
    print(f"  got text for {len(text)}")

    check_trigger_provenance(df, text, findings)
    check_verbosity_confound(df, text, findings)
    check_trigger_logic(findings)
    check_split_and_dedup(df, findings)

    out = pd.DataFrame(findings)
    dest = REPORT_DIR / "label_audit.csv"
    out.to_csv(dest, index=False)

    print("\n" + "=" * 78)
    print("AUDIT SUMMARY")
    print("=" * 78)
    for f in findings:
        mark = "FAIL" if f["verdict"] == "FAIL" else "pass"
        print(f"  [{mark}] {f['check']:<26} {f['detail']}")
    n_fail = sum(f["verdict"] == "FAIL" for f in findings)
    print(f"\n  {n_fail}/{len(findings)} checks failed -> {dest}")


if __name__ == "__main__":
    main()
