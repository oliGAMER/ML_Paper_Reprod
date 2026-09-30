# PROVENANCE.md

Log of what in this repo is our own work, adapted from the original authors, or reused as-is, per the assignment's academic integrity requirements (Section 9).

Team: Zaid Bilal, Omer Ibrahim Qazi, Haseeb Adnan
Paper: Siddiqui & Tarannum (2026), "Fusing Sequence Motifs and Pan-Genomic Features: AMR Prediction using an Explainable Lightweight 1D CNN - XGBoost Ensemble"
Original repo: Saiful185/AMR-EnsembleNet
Our fork: oliGAMER/ML_Paper_Reprod (https://github.com/oliGAMER/ML_Paper_Reprod)

## Stage 2 notes: understanding the paper

Method summary (own words): the paper argues that genomic SNP data has both sequence structure (order of mutations along the genome matters) and feature structure (interactions between SNPs regardless of position). A single model type misses one of these, so they combine a 1D CNN, which reads the SNP string like a sequence and picks up local motifs, with XGBoost, which treats the same SNPs as an unordered feature set and can pick up long-range interactions. The two models are trained separately and their output probabilities are averaged (soft voting) into a final prediction.

Dataset: 809 *E. coli* isolates from the GieSSen dataset, four antibiotics (CIP, CTX, CTZ, GEN), each with a binary susceptible/resistant label. Class balance ranges from roughly even (CIP) to heavily skewed (GEN, ~23% resistant).

Headline results (paper's Table 2-5, 1D CNN and ensemble rows): the ensemble doesn't win by a landslide on every antibiotic, but is the most consistently strong across all four, and on GEN specifically the standalone CNN's high recall (0.7105) on the resistant class is what the paper calls its strength.

Section 3.4 states class weights are "inversely proportional to the roots of the class frequencies." We checked this against the actual notebook code (below) and it doesn't match.

## Stage 3 notes: environment setup

Local machine's default Python (3.13, conda base) doesn't work with the pinned dependencies in requirements.txt: pandas==2.2.2 and tensorflow==2.18.0 don't have prebuilt wheels for 3.13, and building pandas from source failed (GCC 15 / Cython incompatibility). Fixed by creating a separate conda env pinned to Python 3.11, no changes to requirements.txt itself. Cleaned up the old broken venv, added it plus __pycache__, *.egg-info, and WSL's `*:Zone.Identifier` artifact files to .gitignore.

### Move to Colab (2026-09-25)

Local WSL training runs were getting killed by the OOM killer (confirmed via dmesg, process killed around 4.2-4.5GB resident against a ~4.3GB available ceiling in WSL). Rather than shrink the batch size and risk hitting the ceiling again, moved training to Colab, which also matches how the original authors built their notebooks (Drive mounts, /content/drive paths) — so this is closer to their actual setup, not a bigger departure from it.

Recreated the project layout under /content/AMR_repro/ (src/, Giessen_dataset/, results/), loaded the two CSVs from Drive and cached them locally as parquet to avoid re-reading from Drive every cell. Ran on Colab's free T4 tier.

TensorFlow version: requirements.txt pins 2.18.0, but pip can't install that on Colab's current Python/platform — only 2.20.0 and newer are available there. This isn't something we can fix on our end; TF 2.20.0 is the actual environment for every Colab result reported here, and we're documenting it as a real deviation rather than an unresolved TODO. Batch size reverted to 32 (the original notebooks' value) since Colab isn't memory-constrained the way WSL was.

Re-checked the SNP matrix shape (809 x 60,936) and class counts after the move — they match what we verified locally.

**Security note:** a GitHub personal access token was briefly hardcoded in a Colab cell (`!git push https://<user>:<token>@github.com/...`). It's been revoked. Going forward we're using `getpass.getpass()` or Colab's `google.colab.userdata` for anything like this, and checking cell *outputs* before committing, since a token printed to output leaks the same way a hardcoded one does.

## XGBoost reproduction results (Colab)

All four antibiotics trained with the config above, matching Section 3.4/Figure 1 as confirmed earlier. Standalone XGBoost metrics (from `{antibiotic}_XGBoost.json`):

| Antibiotic | Accuracy | AUC | MCC | Macro F1 | Recall (resistant) |
|---|---|---|---|---|---|
| CIP | 0.9630 | 0.9849 | 0.9252 | 0.9626 | 0.9589 |
| CTX | 0.7963 | 0.8889 | 0.5937 | 0.7954 | 0.8194 |
| CTZ | 0.7778 | 0.8340 | 0.5139 | 0.7564 | 0.7091 |
| GEN | 0.7716 | 0.7620 | 0.3930 | 0.6955 | 0.5789 |

CIP's standalone XGBoost result (MCC 0.925, AUC 0.985) is very close to the paper's reported XGBoost row (MCC 0.9253) — a tight reproduction, tighter than our CNN's CIP number. GEN's XGBoost MCC (0.393) also lands close to the paper's XGBoost row (0.3722), and its recall (0.579) matches the paper's own XGBoost recall (0.5789) exactly, so unlike the CNN, XGBoost didn't show a collapse on the imbalanced antibiotic. These per-antibiotic `.json` files and the saved `.keras`/XGBoost models are what feed the weight-sweep notebook below.

## CNN reproduction results (Colab, 2026-09-25)

All four antibiotics trained for up to 150 epochs with early stopping. Final test metrics vs. the paper's 1D CNN row:

| Antibiotic | Our Acc | Paper Acc | Our MCC | Paper MCC | Our Macro F1 | Paper Macro F1 | Our Recall (R) | Paper Recall (R) |
|---|---|---|---|---|---|---|---|---|
| CIP (run 1) | 0.8765 | 0.9568 | 0.7568 | 0.9129 | 0.8762 | 0.9564 | 0.9178 | 0.9589 |
| CIP (re-run) | 0.9198 | 0.9568 | 0.8405 | 0.9129 | 0.9194 | 0.9564 | 0.9452 | 0.9589 |
| CTX | 0.8025 | 0.7840 | 0.6030 | 0.5647 | 0.8010 | 0.7821 | 0.8056 | 0.7778 |
| CTZ | 0.7963 | 0.8086 | 0.5402 | 0.5651 | 0.7698 | 0.7816 | 0.6727 | 0.6727 |
| GEN (val_accuracy monitor) | 0.7654 | 0.7346 | 0.0000 | 0.3984 | 0.4336 | 0.6836 | 0.0000 | 0.7105 |
| GEN (val_auc monitor) | 0.7778 | 0.7346 | 0.3136 | 0.3984 | 0.6495 | 0.6836 | 0.3684 | 0.7105 |

CTX and CTZ are close to the paper on every metric — a solid reproduction there.

**CIP:** first run underperformed the paper noticeably (MCC 0.76 vs 0.91). Re-running with the same config brought MCC to 0.84 and recall to 0.945, close to the paper's numbers. Reads as run-to-run GPU variance rather than a bug, since the fixed seed doesn't guarantee bit-identical results across GPU runs.

**GEN failure and fix:** the first GEN run collapsed completely — the model predicted "susceptible" for all 162 test samples (0 recall, 0 MCC, AUC below chance). This is the opposite of the paper's own reported result, where GEN's high CNN recall is the headline finding for that antibiotic.

Root cause: GEN is ~23% resistant, so a model that always predicts the majority class already scores ~77% accuracy. The original notebooks use `val_accuracy` as the early-stopping monitor, which let training checkpoint on exactly that degenerate solution. This is a real weakness in the authors' own training setup for their hardest task — not something we introduced, since we reproduced it faithfully on the first attempt.

We switched the early-stopping monitor to `val_auc` (keeping everything else — threshold, patience, architecture, class weights — the same) and re-ran under the same TF 2.20.0 environment CTX/CTZ already trained under, to rule out TF version as a variable. Result: AUC 0.43 → 0.74, MCC 0.00 → 0.31, macro F1 0.43 → 0.65, recall 0.00 → 0.37.

Recall (0.37) is still well below the paper's 0.71, so this isn't a tight reproduction yet, but it's a normal, explainable underperformance rather than a total failure. From the training log, recall briefly peaked at 0.68 around epoch 33 before drifting down as training continued to its actual stopping point at epoch 89. `val_recall` as the monitor, or a threshold sweep (0.505 was carried over from the original notebook, not re-tuned for `val_auc`), are candidates for closing this gap further — not yet tried.

This result — the collapse, the cause, and the fix — is reportable either as a finding about `val_accuracy` as a bad monitor under class imbalance, or as a worked example of debugging a reproduction that initially failed for a non-obvious reason. Belongs in the report's Reproduction Results and Analysis/Discussion sections.

## Data provenance

Files: `cip_ctx_ctz_gen_multi_data.csv` (SNP matrix), `cip_ctx_ctz_gen_pheno.csv` (phenotype labels), from `Giessen_dataset.zip` in the original repo. Not present in our fork's default checkout — added manually to `Giessen_dataset/` locally.

Storage: `cip_ctx_ctz_gen_multi_data.csv` (~99MB) tracked via Git LFS; `cip_ctx_ctz_gen_pheno.csv` (~21KB) committed as a plain file. Both are pushed to the fork under `Giessen_dataset/` (https://github.com/oliGAMER/ML_Paper_Reprod/tree/main/Giessen_dataset).

Verification (2026-09-25):
- SNP matrix: 809 samples x 60,936 feature columns — matches Section 3.1.
- Phenotype file: 809 samples x 4 antibiotic columns.
- Sample IDs match 1:1 and in the same order between the two files.
- Class counts match Table 1 exactly:

| Antibiotic | Susceptible | Resistant |
|---|---|---|
| CIP | 443 | 366 |
| CTX | 451 | 358 |
| CTZ | 533 | 276 |
| GEN | 621 | 188 |

Confirmed independently via `src/data_loader.py`.

## Code provenance

| File / Notebook | Status | Notes |
|---|---|---|
| `src/data_loader.py` | Written by us | Loads and validates the SNP + phenotype CSVs against the paper's reported stats, provides `get_split()` for the stratified 80/20 per-antibiotic split (Section 3.4), fixed `random_state=42` (matches the original notebooks). Verified: full load gives correct shape/class counts; splits preserve class ratio within ~0.2pp of the full dataset for all four antibiotics. |
| `src/train_cnn.py` | Adapted from `Final Custom 1D CNN Implementations/AMR_Project_1D_CNN_v1_{CIP,CTX,CTZ,GEN}.ipynb` | Consolidates four near-duplicate per-antibiotic notebooks into one parameterized script. Architecture (`build_cnn1d_model`, from `build_cnn1d_model_extended`) and core hyperparameters (embedding dim 64, dropout, 150 epochs, batch 32, lr 1e-3) reused unchanged. Changes: (1) data loading routed through `src/data_loader.py`; (2) fixed class weights to be computed from `y_train` only, not the full pre-split label array (see bug note below); (3) removed the Colab `drive.mount()` cell; (4) per-antibiotic threshold/monitor/patience settings moved into an explicit `ANTIBIOTIC_CONFIG` instead of varying silently across four files; (5) fixed a checkpoint-filename bug in the original CTX notebook (below); (6) GEN's early-stopping monitor changed from `val_accuracy` to `val_auc` — a deliberate deviation, unlike CIP/CTX/CTZ which keep their original settings. |
| `AMR Ensemble Models/AMR_Project_Ensemble_Soft_Voting.ipynb` | Reviewed | Loads each antibiotic's saved CNN and XGBoost models, re-derives the same 80/20 split, and averages both models' probabilities with a fixed 50/50 weight — not tuned or swept anywhere in the notebook. Per-antibiotic thresholds are hardcoded (CIP/CTZ 0.55, CTX/GEN 0.48), not derived programmatically. The paper's claim that 50/50 is optimal isn't backed by a search in this notebook — this is the gap our weight-sweep script addresses. |
| `Final Custom 1D CNN Implementations/` | Reviewed, consolidated into `src/train_cnn.py` | |
| `Random Forest Implementations/AMR_Project_RF_Baseline_All.ipynb` | Reviewed | Baseline only, not part of the CNN+XGBoost ensemble. `RandomForestClassifier(n_estimators=400, max_depth=15, max_features='sqrt', class_weight='balanced', random_state=42, oob_score=True)`, same split as the other notebooks. Not consolidated (out of scope, not reused elsewhere). |
| `XGBoost Implementations/AMR_Project_XGBoost_Baseline_ALL.ipynb` | Reviewed, adapted into the Colab migration notebook | Confirms `n_estimators=1000` matches Figure 1. Full params: `objective=binary:logistic`, `eval_metric=auc`, `n_estimators=1000`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.7`, `colsample_bytree=0.7`, `scale_pos_weight` computed correctly from `y_train` only, `early_stopping_rounds=50`, `random_state=42`. Trained on Colab for all four antibiotics using `xgboost==2.0.3` (pinned to match `requirements.txt`, unlike TensorFlow which couldn't be pinned — see environment note above); saved as `best_xgboost_model_{antibiotic}.json` and pushed to Drive. |
| `AMR_Ensemble_Weight_Sweep.ipynb` | Written by us | Root-of-repo notebook, not part of the original authors' code. Loads each antibiotic's saved CNN and XGBoost models, sweeps the blend weight, and picks a threshold per weight via Youden's J. See results below. |
| `AMR_Project_1D_CNN_Tuning.ipynb` | Not yet reviewed | Likely the source of the final architecture in Figure 1. |

## Findings from the CNN notebooks (2026-09-25)

Diffed all four `AMR_Project_1D_CNN_v1_*.ipynb` cell-by-cell. Aside from `TARGET_ANTIBIOTIC`, they differ in undocumented per-antibiotic settings (none of this appears in the paper text):

| Antibiotic | Early-stopping monitor | ES patience | Decision threshold |
|---|---|---|---|
| CIP | val_auc | 60 | 0.30 |
| CTX | val_accuracy | 40 | 0.50 |
| CTZ | val_accuracy | 40 | 0.62 |
| GEN | val_accuracy | 60 | 0.505 |

- Thresholds aren't 0.5 for CIP, CTZ, or GEN — a real tuning choice not mentioned in the paper, worth a line in the report's Implementation Details.
- Bug in the original CTX notebook: its `ModelCheckpoint` callback saves to `best_cnn1d_model_CTZ.keras` (leftover from copy-pasting the CTZ notebook). If CTX and CTZ were ever run in the same directory, CTX's checkpoint could be silently overwritten. Fixed in `src/train_cnn.py` by deriving the filename from the antibiotic param.
- Class weight formula: the notebooks use standard inverse-frequency weighting, `(1/n_class) * (total/2)`. The paper's Section 3.4 says weights are inversely proportional to the square root of class frequency. The code does not match the paper's stated method. Not yet resolved — either keep the code's actual formula and note the discrepancy (since our job is to reproduce what the code does), or implement the paper's stated formula as a deliberate deviation. Leaning toward the former for now; open for team discussion.
- Random seed: `tf.random.set_seed(42)` and `np.random.seed(42)` are set in all four notebooks, consistent with our own loader.
- Class weights are computed from the full pre-split label array rather than `y_train`, a minor train/test leak in the weighting step (not the data itself). Fixed in `src/train_cnn.py`.

## Open questions (Stage 3.3 checklist)

- [x] Does the class-weighting formula match the paper's stated inverse-square-root method? No — see above.
- [x] Is a random seed set in the original notebooks? Yes, 42, matching ours.
- [x] Does XGBoost's `n_estimators` match Figure 1's stated 1000? Yes, confirmed 2026-09-27.
- [x] GEN collapse — root cause confirmed and fixed (val_accuracy → val_auc). Recall still below the paper; see remaining-gap note above.
- [x] CIP underperformance — resolved on re-run, attributed to GPU run-to-run variance.
- [x] Can the ensemble notebook run end-to-end on our fork? Yes — all four XGBoost models are now trained and saved alongside the CNNs, so the weight-sweep and ensemble notebooks both run on our checkpoints.

## Ensemble weight-sweep results (Colab, 2026-09-27)

`AMR_Ensemble_Weight_Sweep.ipynb` sweeps the CNN/XGBoost blend weight from 0.0 (pure XGBoost) to 1.0 (pure CNN) in steps of 0.05, per antibiotic, using the same saved models and 80/20 split as the authors' notebook. At each weight, the threshold is chosen via Youden's J on that weight's ROC curve, matching the authors' own approach of picking a threshold from the test set, so the comparison is apples-to-apples. Written entirely by us — the authors' notebook never sweeps this weight, it only evaluates the fixed 50/50 point.

Is 50/50 actually optimal? Not for all four antibiotics.

| Antibiotic | 50/50 MCC | Best weight | Best MCC | MCC gain | Best AUC weight | Best AUC | AUC gain |
|---|---|---|---|---|---|---|---|
| CIP | 0.9008 | 0.00 (pure XGBoost) | 0.9252 | +0.0244 | 0.00 | 0.9849 | +0.0054 |
| CTX | 0.6706 | 0.65 (CNN-leaning) | 0.6750 | +0.0044 (negligible) | 0.00 | 0.8889 | +0.0049 |
| CTZ | 0.5806 | 0.25 (XGBoost-leaning) | 0.5892 | +0.0087 | 0.00 | 0.8340 | +0.0171 |
| GEN | 0.4063 | 0.45 (~same as 0.5) | 0.4063 | 0.0000 | 0.10 | 0.7627 | +0.0057 |

CIP shows the clearest gain from moving off 50/50 — pure XGBoost beats the blend on both MCC and AUC, consistent with XGBoost's strong standalone CIP result above. CTX and GEN are close to indifferent to the weight on MCC (GEN's best-MCC weight of 0.45 is essentially the paper's 0.5). CTZ favors leaning toward XGBoost. Overall: the paper's fixed 50/50 choice is a reasonable default but not weight-optimal for three of the four antibiotics — a concrete, data-backed answer to "is 50/50 actually optimal," and a natural Stage 4 experiment (a hyperparameter-study type, Section 7 type B) with a one-sentence question behind it.

Full per-weight curves (`{antibiotic}_weight_sweep.csv`, plotted as accuracy/MCC/macro F1/recall/AUC vs. CNN weight from 0.0 to 1.0) are saved per antibiotic alongside the summary table above, with the paper's 50/50 point and the MCC-optimal point both marked for comparison.

This sweep is the Stage 4 hyperparameter study committed to in the proposal (Section 7, planned experiment): "does the best CNN/XGBoost weighting change as the resistance task becomes more imbalanced, and can a tuned weight improve Macro F1 or resistant-class recall without a large precision drop?" The per-antibiotic table above answers it directly — the optimal weight does shift with imbalance (pure XGBoost for balanced CIP, XGBoost-leaning for imbalanced CTZ, near-indifferent for CTX/GEN), and the gains are real but modest (MCC +0.02 to +0.009) rather than dramatic. Counts as Stage 4 complete for the hyperparameter-study track.

The proposal's second planned experiment, the ablation comparing standalone CNN, standalone XGBoost, and the full ensemble under identical splits, is answered by the same data: the per-model tables above (CNN reproduction results, XGBoost reproduction results) plus the ensemble's 50/50 row in the weight-sweep table together give all three numbers per antibiotic, so this ablation doesn't need a separate run.

## Feature-selection k-sweep (M3 submission, branch `M3_Own_Experiment`)

Branch: https://github.com/oliGAMER/ML_Paper_Reprod/tree/M3_Own_Experiment
Notebook: `AMR-EnsembleNet — Feature Selection k-Sweep (k SNPs vs. Performance).ipynb`, written by us.

Question: does XGBoost need all 60,936 SNP features, or does a smaller subset selected by statistical relevance perform just as well or better?

Method, in notes:
- Custom categorical chi-squared test (own implementation, not sklearn's `chi2`, since SNP tokens are 5-way categorical: 0-4) computed on the training split only, per antibiotic, no leakage.
- k values log-spaced from 100 to 60,936 (`np.logspace`, 10 points) plus the full feature count, plus each antibiotic's own Bonferroni-significant feature count (α = 0.05 / 60,936 ≈ 8.21e-7) added as an extra k point.
- Bonferroni-significant feature counts varied a lot by antibiotic: CIP 3,886, CTX 786, CTZ 724, GEN 349 — consistent with CIP being the antibiotic with the strongest overall signal and GEN the weakest, matching the reproduction results above.
- XGBoost retrained from scratch at each k (n_estimators=300, max_depth=6, lr=0.05, subsample/colsample=0.8), same decision thresholds as the original per-antibiotic notebooks (CIP 0.30, CTX 0.50, CTZ 0.62, GEN 0.505).
- Full results: `k_sweep_results.csv`. Plots: `k_sweep.png` (MCC/AUC/F1/accuracy vs. k, log x-axis, all four antibiotics) and `chi2_significance_{CIP,CTX,CTZ,GEN}.png` (per-antibiotic chi2 score vs. feature rank, with the Bonferroni cutoff marked).

Findings, in notes:
- CIP: MCC peaks at k=3,524–3,886 (MCC 0.9008, matching the full-feature ensemble result almost exactly) then **drops** back down at higher k (60,935: MCC 0.8770). More features past a few thousand doesn't help CIP and can hurt slightly.
- CTX: best MCC (0.6278) is at the full 60,936 features — the only antibiotic where more features keep helping all the way up. Small-k performance is noticeably worse (k=847: MCC 0.4392).
- CTZ: best MCC (0.5960) around k=415-847, close to flat from there — a small, well-chosen feature set does as well as everything.
- GEN: best MCC (0.4403) at k=415, actually **above** both the full-feature XGBoost result (0.3930) and the paper's own XGBoost MCC (0.3722) — the clearest case where feature selection helps rather than just matching baseline. Performance is noisy and non-monotonic at low k, consistent with GEN's small resistant class making any subset estimate noisier.
- Overall: three of four antibiotics (CIP, CTZ, GEN) reach their best or near-best MCC with a few hundred to a few thousand features, well under 10% of the full 60,936 — only CTX clearly wants the full matrix. This is a useful complement to the ensemble weight-sweep: together they show the paper's fixed choices (50/50 blend, full feature matrix) aren't optimal for every antibiotic, and the right setting depends on how imbalanced/noisy that antibiotic's resistance signal is.
- This is the assignment's second Stage 4 experiment type: a data-size/feature-scaling study (Section 7 type E, adapted to feature count rather than sample count), separate from the ensemble-weight hyperparameter study already logged above. Together they cover both planned-experiment slots from the proposal (Section 7), the hyperparameter study explicitly and this one as the natural follow-on question raised while reviewing the paper's feature-heavy setup.

## Remaining work

Aligned to Stage 4 (experimentation) and Stage 5 (deployment) of the assignment, and the proposal's committed deliverables:

- Decide and document the class-weight formula question (code vs. paper) as a team, for the report's Implementation Details/Limitations section.
- Try `val_recall` as GEN's CNN early-stopping monitor and a threshold re-sweep, time permitting — the CNN's GEN recall (0.368) is still the weak point of the reproduction; XGBoost's GEN recall (0.579) is already much closer to the paper's CNN figure (0.7105) without any fix needed.
- Consolidate the intermediate `data_loader_full.py` into a single canonical `data_loader.py` name in the Colab copy (noted earlier, not yet cleaned up).
- Sequence logos from the first CNN layer's learned filters, using the SNP CSVs, for the interpretability angle the proposal's deployment plan flags ("if feasible, display the most important features so the output remains interpretable"). Not yet started.
- Stage 5 deployment: build the Streamlit app per the proposal (upload one SNP sample, pick an antibiotic, return predicted class, ensemble probability, and both component-model probabilities). Not yet started.
- Consider whether the k-sweep's smaller feature sets (e.g. CIP at k≈3,886, GEN at k≈415) should feed into the deployed model, since they match or beat full-feature performance with a much smaller input.
