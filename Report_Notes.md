# Report notes — AMR-EnsembleNet reproduction

Note form, for turning into the actual report. Source detail in PROVENANCE.md / README.md fork-notes section.

## Introduction and problem statement
- AMR (antimicrobial resistance) prediction from bacterial genomic data — practical problem in surveillance/medicine.
- Standard ML treats SNPs as an unordered feature bag, loses order/local-motif info.
- Sequence models (transformers) too data-hungry for datasets this size.
- Paper's angle: combine both views — a CNN for order/motifs, XGBoost for unordered feature interactions.

## Literature/background
- Prior work either sequence-based (misses feature interactions, needs lots of data) or feature-based/tree-based (misses order).
- Ensemble/soft-voting is the paper's answer — not novel as a technique, novel as applied here to fuse these two specific model families on genomic SNP data.
- TODO: 2-3 cited works + 1-2 citing works, per Stage 2 Pass 4 — not yet done.

## Paper methodology
- Dataset: GieSSen, 809 *E. coli* isolates, 60,936 SNP features, 4 antibiotics (CIP/CTX/CTZ/GEN), binary resistant/susceptible.
- 1D CNN: embedding dim 64, 6 conv blocks, pooling, global max pool, small MLP head.
- XGBoost: n_estimators=1000, lr=0.05, max_depth=6, subsample/colsample=0.7, scale_pos_weight from y_train.
- Ensemble: soft voting, fixed 50/50 average of CNN + XGBoost probabilities.
- Class weights stated as inverse-sqrt of class frequency (Section 3.4) — but code actually uses standard inverse frequency. Discrepancy, not yet resolved which to treat as "the method."
- Per-antibiotic early-stopping monitor/patience/threshold all vary and are undocumented in the paper text (see Implementation details).

## Implementation details
- Env: conda, Python 3.11 (3.13 broke pandas/TF builds). Moved to Colab after WSL OOM-killed local runs.
- TF pinned to 2.20.0 not 2.18.0 (requirements.txt version unavailable on Colab) — documented deviation.
- XGBoost pinned to 2.0.3, matches requirements.txt.
- data_loader.py: own code, stratified 80/20 split per antibiotic, seed 42 (matches original notebooks).
- train_cnn.py: consolidates 4 near-duplicate per-antibiotic notebooks into one script. Fixed: (a) class-weight leak (was computed pre-split on full labels, now train-only), (b) CTX checkpoint-filename bug (was overwriting/reading CTZ's file).
- GEN's early-stopping monitor deliberately changed val_accuracy → val_auc (see Reproduction results / Analysis below for why).
- Decision thresholds per antibiotic taken from original notebooks (not 0.5 default): CIP 0.30, CTX 0.50, CTZ 0.62, GEN 0.505.
- Class-weight formula: kept the code's actual formula (inverse frequency) rather than the paper's stated one (inverse sqrt) for the reproduction — team decision still pending on how to frame this in the report.

## Dataset and preprocessing
- 809 samples × 60,936 SNP columns, verified shape and class counts match paper's Table 1 exactly.
- Class balance: CIP 443/366 (near-balanced), CTX 451/358, CTZ 533/276, GEN 621/188 (most imbalanced, ~23% resistant).
- SNP tokens: categorical, 5 states (0-4).
- Sample IDs matched 1:1 in order between SNP matrix and phenotype files.

## Reproduction results
- CNN: CTX/CTZ close to paper on all metrics. CIP needed a re-run (GPU variance) to close gap, landed close after. GEN collapsed entirely on first run (predicted majority class for every test sample) under val_accuracy monitor — root cause: ~23% resistant means "always predict susceptible" already scores ~77% accuracy, so val_accuracy is a bad stopping signal. Fixed via val_auc monitor; closed most but not all of the gap (recall still 0.368 vs paper's 0.7105).
- XGBoost: trained clean for all four, no collapse. CIP essentially matches paper's XGBoost row. GEN's XGBoost recall (0.579) matches the paper's own XGBoost recall exactly, and is much healthier than the CNN's GEN result — XGBoost didn't have the CNN's imbalance problem.
- Numbers: see tables in README fork-notes / PROVENANCE.md.

## Additional experiment
Two planned/run, matching the proposal's Section 7 commitments:
1. **Ensemble weight sweep** (hyperparameter study). Swept CNN/XGBoost blend weight 0.0-1.0. Answer: 50/50 not optimal for 3/4 antibiotics — CIP favors pure XGBoost, CTZ favors XGBoost-leaning, CTX/GEN near-indifferent. Gains modest (MCC +0.004 to +0.024).
2. **Feature-selection k-sweep** (feature-scaling study, branch `M3_Own_Experiment`). Chi-squared feature selection, log-spaced k from 100 to 60,936 + Bonferroni-significant count per antibiotic. Answer: CIP/CTZ/GEN all reach best or near-best MCC with a few hundred to a few thousand features (<10% of full set); only CTX wants the full matrix. GEN's k=415 result (MCC 0.4403) beats both this fork's full-feature XGBoost (0.3930) and the paper's reported XGBoost (0.3722) — feature selection genuinely helps on the hardest antibiotic.
- Both experiments together cover the proposal's committed hyperparameter-study and ablation/scaling experiment slots. The CNN-vs-XGBoost-vs-ensemble ablation is also answered as a byproduct (per-model tables + ensemble's 50/50 row give all three numbers per antibiotic).

## Analysis and discussion
- GEN collapse is the most interesting finding: a real weakness in the original authors' training setup (not something introduced), reproduced faithfully, then fixed and explained. Worth framing as a worked debugging example, not just a number.
- Why the CNN and XGBoost diverge so much on GEN: CNN more sensitive to class imbalance via its stopping criterion; XGBoost's scale_pos_weight handles imbalance more directly, hence healthier GEN recall.
- k-sweep result reframes the imbalance story: fewer, more relevant features actually help GEN more than a fixed 50/50 ensemble blend does — feature selection may matter more than ensemble weighting for the hardest antibiotic.
- Class-weight formula mismatch (paper says inverse-sqrt, code does inverse-frequency): flag as authors' own inconsistency between text and code, reproduced the code's actual behavior since that's what generates the reported numbers.
- CIP run-to-run variance (first run underperformed, re-run closed the gap): worth a line on reproducibility limits of single-seed runs even with a fixed random_state, since GPU non-determinism isn't fully controlled by the seed alone.

## References
- Siddiqui, M. S. B., & Tarannum, N. (2026). Fusing Sequence Motifs and Pan-Genomic Features: Antimicrobial Resistance Prediction using an Explainable Lightweight 1D CNN-XGBoost Ensemble. SCA/HPCAsia 2026. ACM. DOI: 10.1145/3773656.3773682.
- Original repo: Saiful185/AMR-EnsembleNet.
- GieSSen dataset via ML-iAMR: YunxiaoRen/ML-iAMR.
- Our fork: oliGAMER/ML_Paper_Reprod (main branch + M3_Own_Experiment branch for the k-sweep).
- TODO: 2-3 prior-work citations, 1-2 citing-work citations (Stage 2 Pass 4, not yet done).

## Still open / not yet done
- Literature review pass (2-3 cited + 1-2 citing papers) — not started.
- Team decision on class-weight formula framing for the report.
- val_recall monitor + threshold re-sweep for GEN CNN, time permitting.
- Sequence logos from CNN first-layer filters (interpretability, ties to proposal's deployment plan).
- Streamlit deployment (Stage 5) — not started.
- Random Forest baseline — reviewed only, not run on this fork.
