# AMR-EnsembleNet Deployment

A Streamlit app that serves two models for predicting antimicrobial
resistance from SNP data. Pick an antibiotic, a feature set, and a held-out
test isolate; the app returns the model's probability(ies), the predicted
label, and the real ground-truth label for that isolate.

**Two feature-set modes:**
- **Full (60,936 features)** — the 1D CNN + XGBoost soft-voting ensemble from
  Sections 4 and 6 of the report. Returns the CNN probability, the XGBoost
  probability, and the blended ensemble probability.
- **Optimized (chi-squared selected)** — an XGBoost-only model (no CNN)
  trained on a much smaller, statistically selected subset of features, from
  the feature-selection k-sweep in Section 7.2 of the report.

## How to start the app

From the repository root:

```bash
cd ML_Paper_Reprod/deployment
pip install -r requirements.txt
streamlit run app.py
```

This opens the app at `http://localhost:8501`.

**Windows note:** installing `tensorflow` can fail with an `OSError` on paths
like `...\tensorflow\include\external\...` if Windows Long Path support is
off. If `pip install` fails that way, create the virtual environment at a
short path (e.g. `C:\tfenv`) instead of inside a deeply nested project
folder, and install there:

```bash
python -m venv C:/tfenv
C:/tfenv/Scripts/pip install -r requirements.txt
C:/tfenv/Scripts/streamlit run app.py
```

## Dependencies

See `requirements.txt` in this folder (`streamlit`, `pandas`, `numpy`,
`scikit-learn`, `tensorflow`, `xgboost`, pinned to the same versions as the
main project's `requirements.txt` plus `streamlit`).

## Input format

The app does not take free-form input. SNP vectors are 60,936 features per
isolate, which isn't something a user can type by hand, so instead:

1. Pick an **antibiotic**: `CIP`, `CTX`, `CTZ`, or `GEN`.
2. Pick a **feature set**: Full (60,936 features) or Optimized (chi-squared
   selected). Optimized feature counts: CIP 3,524; CTX 60,936 (same as full —
   see note below); CTZ 415; GEN 415.
3. Pick a **test isolate ID** from a dropdown. The dropdown only lists the 162
   isolates in that antibiotic's held-out 20% test split (reconstructed with
   the same `train_test_split(..., stratify=y, random_state=42)` call used
   everywhere else in the project) — so every isolate shown was never used to
   train the models being queried.

Internally, the app loads that isolate's real SNP row from
`Giessen_dataset/cip_ctx_ctz_gen_multi_data.csv` and its real label from
`Giessen_dataset/cip_ctx_ctz_gen_pheno.csv`; nothing is entered manually.

## Output format

After clicking **Predict**, the app displays, for **Full** mode:

| Field | Meaning |
|---|---|
| CNN probability (resistant) | Sigmoid output of the per-antibiotic 1D CNN |
| XGBoost probability (resistant) | `predict_proba` output of the per-antibiotic XGBoost model |
| Ensemble probability (resistant) | `0.5 * CNN + 0.5 * XGBoost` |
| Prediction | `Resistant` if the ensemble probability ≥ the paper's fixed threshold for that antibiotic (0.55 for CIP/CTZ, 0.48 for CTX/GEN), else `Susceptible` |
| Ground truth | The isolate's real lab-measured phenotype label, with a match/mismatch indicator |

or, for **Optimized** mode:

| Field | Meaning |
|---|---|
| XGBoost probability (resistant) | `predict_proba` output of the reduced-feature XGBoost model |
| Features used | `k` out of 60,936, and the percentage that represents |
| Prediction | `Resistant` if the probability > the k-sweep's own threshold for that antibiotic (0.30 CIP, 0.50 CTX, 0.62 CTZ, 0.505 GEN — these are different thresholds than Full mode's ensemble thresholds) |
| Ground truth | Same as above |

## Optimized-feature model: provenance and a known caveat

The optimized models (`results/best_xgboost_optimized_{AB}.json`) and their
selected feature indices (`results/optimized_feature_indices_{AB}.json`) were
retrained specifically for this deployment, using the exact method from the
k-sweep notebook (`AMR_EnsembleNet_—_Feature_Selection_k_Sweep_...ipynb`):
the same custom chi-squared selector (computed on the training split only),
the same XGBoost hyperparameters (300 trees, no early stopping), and the
antibiotic's best-MCC `k` from `m3/k_sweep_results.csv`.

**Caveat, found and kept rather than hidden:** retraining reproduced CTZ and
GEN's published k-sweep metrics almost exactly, but CIP and CTX came out
close, not identical (CIP: MCC 0.877 here vs. 0.901 published; CTX: MCC 0.605
here vs. 0.628 published). The most likely cause is an XGBoost version
difference between whatever ran on Colab for the original k-sweep and the
`xgboost==3.2.0` installed locally for this deployment (the project's
`requirements.txt` pins `2.0.3`, but installing that version locally broke
model saving outright, with an unrelated internal API error, so `3.2.0` was
kept instead). The deployed Optimized-mode numbers above are this retrained
model's own, freshly and correctly computed metrics — they are real and
internally consistent, just not bit-identical to the `k_sweep_results.csv`
numbers already cited in the report.

## One worked example, end to end

Antibiotic: **CIP**, isolate: **H108_S5_L001** — captured live from the running
app (see screenshot below).

| Field | Value |
|---|---|
| CNN probability | 0.084 |
| XGBoost probability | 0.009 |
| Ensemble probability | 0.046 |
| Threshold (CIP) | 0.55 |
| Prediction | Susceptible |
| Ground truth | Susceptible |
| Match | Yes |

A second, script-reproduced example (antibiotic CIP, isolate
`RBL-17-0409-1_S127`): CNN 0.023, XGBoost 0.015, ensemble 0.019, prediction
Susceptible, ground truth Susceptible, match — reproduced by directly calling
the same `get_test_split` / `predict_full` functions `app.py` uses, outside
Streamlit.

**Optimized mode** (script-verified, not yet confirmed in the browser UI):
antibiotic CTZ, isolate `RBT-17-0285-1_S66` — XGBoost probability 0.747,
threshold 0.62, prediction Resistant, ground truth Susceptible, **mismatch**
(an honest example, not a cherry-picked correct one).

## Screenshot

![AMR-EnsembleNet Streamlit app showing a CIP prediction for isolate H108_S5_L001: CNN probability 0.084, XGBoost probability 0.009, ensemble probability 0.046, predicted Susceptible, matching the ground truth](screenshot.png)

TODO: capture a second screenshot of **Optimized** mode before final submission
— this mode has only been verified via script so far (see smoke test in this
session), not clicked through in the browser.
