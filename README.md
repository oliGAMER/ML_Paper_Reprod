# AMR-EnsembleNet

Fusing Sequence Motifs and Pan-Genomic Features: Antimicrobial Resistance Prediction using an Explainable Lightweight 1D CNN - XGBoost Ensemble

**Official ScikitLearn/TensorFlow/Keras implementation for the paper: "Fusing Sequence Motifs and Pan-Genomic Features: Antimicrobial Resistance Prediction using an Explainable Lightweight 1D CNN - XGBoost Ensemble"**

> **Summary:** The rapid and accurate prediction of Antimicrobial Resistance (AMR) from genomic data is a critical challenge in modern medicine. Standard machine learning models often treat the genome as an unordered "bag of features," ignoring the valuable sequential information encoded in the order of Single Nucleotide Polymorphisms (SNPs). Conversely, state-of-the-art sequence models like Transformers are often too data-hungry for the moderately-sized datasets typical in genomic surveillance. We propose **AMR-EnsembleNet**, a simple yet powerful ensemble framework that synergistically combines the strengths of these two approaches. Our framework fuses a lightweight, custom-tuned 1D Convolutional Neural Network (CNN), designed to learn predictive sequence motifs, with an XGBoost model adept at capturing complex, non-local feature interactions. When trained and evaluated on a benchmark dataset of 809 *E. coli* isolates, our ensemble model consistently achieves top-tier performance across four antibiotics with varying class imbalance. For the highly challenging Gentamicin (GEN) dataset, the ensemble yields the best overall performance with a Matthews Correlation Coefficient (MCC) of 0.403, a score driven by the CNN's superior ability to recall rare resistant cases. Our results show that fusing complementary sequence-based and feature-based models provides a robust, accurate, and computationally feasible solution for AMR prediction.

---

## Architecture Overview

The AMR-EnsembleNet is a simple soft voting ensemble that combines the prediction probabilities of two powerful, complementary models:

1.  **A Sequence-Aware Custom 1D CNN:** A deep, lightweight 1D Convolutional Neural Network (CNN) processes the ordered sequence of integer-encoded SNPs. Its hierarchical convolutional layers are designed to learn local patterns and motifs, such as multiple functionally related mutations within a single gene.
2.  **A Feature-Based XGBoost Model:** An XGBoost classifier operates on the same SNP data but treats it as an unordered "bag of features." This allows it to learn complex, non-linear interactions between SNPs, regardless of their position on the chromosome.

The final prediction is the unweighted/weighted average of the probabilities from these two models, creating a more robust and generalized classifier.

![AMR-EnsembleNet_Diagram](figures/AMR_EnsembleNet.png) 

---

## Key Results

A central finding of this research is that no single model is universally superior across all AMR prediction tasks. The best performance is often achieved by an ensemble that leverages the complementary strengths of a sequence-aware deep learning model and a powerful tree-based model, especially on challenging, imbalanced datasets.

#### Performance on Ciprofloxacin (CIP) - Balanced Dataset

| Model | Accuracy | F1 Score (Resist.) | Matthews (MCC) | Precision | Recall | F1 Score (Macro) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random Forest | 0.9568 | 0.9517 | 0.9127 | 0.9583 | 0.9452 | 0.9563 |
| XGBoost | **0.9630** | **0.9583** | 0.9253 | 0.9718 | 0.9452 | **0.9625** |
| 1D CNN | 0.9568 | 0.9524 | 0.9129 | 0.9459 | **0.9589** | 0.9564 |
| **AMR-FusionNet (Ensemble)** | **0.9630** | 0.9577 | **0.9260** | **0.9855** | 0.9315 | 0.9624 |

#### Performance on Gentamicin (GEN) - Highly Imbalanced Dataset

| Model | Accuracy | F1 Score (Resist.) | Matthews (MCC) | Precision | Recall | F1 Score (Macro) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random Forest | 0.7469 | 0.4938 | 0.3271 | 0.4651 | 0.5263 | 0.6626 |
| XGBoost | **0.7593** | 0.5301 | 0.3722 | **0.4889** | 0.5789 | 0.6841 |
| 1D CNN | 0.7346 | 0.5567 | 0.3984 | 0.4576 | **0.7105** | 0.6836 |
| **AMR-FusionNet (Ensemble)** | 0.7469 | **0.5591** | **0.4030** | 0.4727 | 0.6842 | **0.6908** |

The 1D CNN's superior recall on the rare resistant class for Gentamicin is a key finding, which the ensemble successfully leverages to achieve the best overall balanced performance (highest MCC and Macro F1-score).

---

## Setup and Installation

This project is built using TensorFlow and Scikit-learn.

**1. Clone the repository:**
```bash
git clone https://github.com/Saiful185/AMR-EnsembleNet.git
cd AMR-EnsembleNet
```

**2. Install dependencies:**
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```
Key dependencies include: `tensorflow`, `scikit-learn`, `shap`, `xgboost`, `pandas`, `numpy`, `matplotlib`.

---

## Dataset

The experiments are run on the publicly available GieSSen dataset. The specific files used are the **SNP-matrix** (`cip_ctx_ctz_gen_multi_data.csv`) and the **phenotype data** (`cip_ctx_ctz_gen_pheno.csv`). These are included in this repository or can be downloaded from the original [ML-iAMR GitHub](https://github.com/YunxiaoRen/ML-iAMR).

## Usage: Running the Experiments

The code is organized into Jupyter/Colab notebooks (`.ipynb`) for each model and the final ensemble.

1.  Open a notebook.
2.  Update the paths in the first few cells to point to your dataset's location.
3.  Run the cells sequentially to perform data setup, model training, and final evaluation. The training notebooks will save the final models, which are then loaded by the ensembling notebook.

---

## Pre-trained Models

The pre-trained model weights for our key experiments are available for download from the [v1.0.0 release](https://github.com/Saiful185/AMR-EnsembleNet/releases/v1.0.0) on this repository.

| Model | Trained For | Description | Download Link |
| :--- | :--- | :--- | :--- |
| **1D CNN** | Ciprofloxacin | Our sequence-aware 1D-CNN model for CIP. | [Link](https://github.com/Saiful185/AMR-EnsembleNet/releases/download/v1.0.0/AMR_cnn1d_model_CIP.keras) |
| **XGBoost** | Ciprofloxacin | The feature-based baseline for CIP. | [Link](https://github.com/Saiful185/AMR-EnsembleNet/releases/download/v1.0.0/AMR_xgboost_model_CIP.json) |
| **Random Forest** | Ciprofloxacin | The Random Forest baseline for CIP. | [Link](https://github.com/Saiful185/AMR-EnsembleNet/releases/download/v1.0.0/AMR_RF_model_CIP.pkl) |
| **1D CNN** | Cefotaxime | Our sequence-aware 1D-CNN model for CTX. | [Link](https://github.com/Saiful185/AMR-EnsembleNet/releases/download/v1.0.0/AMR_cnn1d_model_CTX.keras) |
| **XGBoost** | Cefotaxime | The feature-based baseline for CTX. | [Link](https://github.com/Saiful185/AMR-EnsembleNet/releases/download/v1.0.0/AMR_xgboost_model_CTX.json) |
| **Random Forest** | Cefotaxime | The Random Forest baseline for CTX. | [Link](https://github.com/Saiful185/AMR-EnsembleNet/releases/download/v1.0.0/AMR_RF_model_CTX.pkl) |
| **1D CNN** | Ceftazidime | Our sequence-aware 1D-CNN model for CTZ. | [Link](https://github.com/Saiful185/AMR-EnsembleNet/releases/download/v1.0.0/AMR_cnn1d_model_CTZ.keras) |
| **XGBoost** | Ceftazidime | The feature-based baseline for CTZ. | [Link](https://github.com/Saiful185/AMR-EnsembleNet/releases/download/v1.0.0/AMR_xgboost_model_CTZ.json) |
| **Random Forest** | Ceftazidime | The Random Forest baseline for CTZ. | [Link](https://github.com/Saiful185/AMR-EnsembleNet/releases/download/v1.0.0/AMR_RF_model_CTZ.pkl) |
| **1D CNN** | Gentamicin | Our sequence-aware 1D-CNN model for GEN. | [Link](https://github.com/Saiful185/AMR-EnsembleNet/releases/download/v1.0.0/AMR_cnn1d_model_GEN.keras) |
| **XGBoost** | Gentamicin | The feature-based baseline for GEN. | [Link](https://github.com/Saiful185/AMR-EnsembleNet/releases/download/v1.0.0/AMR_xgboost_model_GEN.json) |
| **Random Forest** | Gentamicin | The Random Forest baseline for GEN. | [Link](https://github.com/Saiful185/AMR-EnsembleNet/releases/download/v1.0.0/AMR_RF_model_GEN.pkl) |

---

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{Siddiqui2026AMREnsembleNet,
	author = {Siddiqui, Md. Saiful Bari and Tarannum, Nowshin},
	title = {Fusing Sequence Motifs and Pan-Genomic Features: Antimicrobial Resistance Prediction using an Explainable Lightweight 1D CNN - XGBoost Ensemble},
	year = {2026},
	isbn = {9798400720673},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	url = {https://doi.org/10.1145/3773656.3773682},
	doi = {10.1145/3773656.3773682},
	booktitle = {Proceedings of the Supercomputing Asia and International Conference on High Performance Computing in Asia Pacific Region},
	pages = {374–383},
	location = {},
	series = {SCA/HPCAsia '26}
}
```

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---
---

# Fork Notes: Reproduction (oliGAMER/ML_Paper_Reprod)

Everything above this line is the original authors' README, kept as-is for comparison. This section documents this fork's own reproduction work for the ML Paper Reproduction & Deployment assignment. Full provenance detail (what's written by us, adapted, or reused as-is) is tracked in `PROVENANCE.md`.

Team: Zaid Bilal, Omer Ibrahim Qazi, Haseeb Adnan

## What's different in this fork

- Environment: local training moved to Google Colab after WSL's OOM killer repeatedly killed training runs (see PROVENANCE.md for details). TensorFlow ended up pinned to 2.20.0 instead of the original 2.18.0, since 2.18.0 has no installable wheel for Colab's current Python/platform — a real, documented environment deviation, not an oversight.
- `src/data_loader.py`: written by us. Loads and validates the SNP + phenotype CSVs against the paper's reported stats, and provides a stratified 80/20 per-antibiotic split matching Section 3.4.
- `src/train_cnn.py`: adapted from the four original per-antibiotic CNN notebooks, consolidated into one parameterized script. Fixes a class-weight leakage bug (weights were computed on the full pre-split label array in the originals, now computed from the training split only) and a checkpoint-filename bug found in the original CTX notebook. GEN's early-stopping monitor was deliberately changed from `val_accuracy` to `val_auc` after diagnosing a training collapse under class imbalance (see PROVENANCE.md).
- `AMR_Ensemble_Weight_Sweep.ipynb`: written by us, not part of the original repo. Sweeps the CNN/XGBoost blend weight instead of using a fixed 50/50 average, to test whether 50/50 is actually optimal per antibiotic.

## Reproduction results (this fork)

### 1D CNN

| Antibiotic | Accuracy | MCC | Macro F1 | Recall (resistant) | Paper MCC | Paper Recall (resistant) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CIP | 0.9198 | 0.8405 | 0.9194 | 0.9452 | 0.9129 | 0.9589 |
| CTX | 0.8025 | 0.6030 | 0.8010 | 0.8056 | 0.5647 | 0.7778 |
| CTZ | 0.7963 | 0.5402 | 0.7698 | 0.6727 | 0.5651 | 0.6727 |
| GEN | 0.7778 | 0.3136 | 0.6495 | 0.3684 | 0.3984 | 0.7105 |

CTX and CTZ are close to the paper on every metric. CIP needed a re-run to close an initial gap attributed to GPU run-to-run variance. GEN initially collapsed entirely under the original `val_accuracy` early-stopping monitor (a real weakness in the original training setup for this most-imbalanced antibiotic, reproduced faithfully) and was fixed by switching to `val_auc`; the fix closes most of the gap but recall (0.368) is still below the paper's 0.7105 — see PROVENANCE.md for the full root-cause writeup.

### XGBoost

| Antibiotic | Accuracy | AUC | MCC | Macro F1 | Recall (resistant) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CIP | 0.9630 | 0.9849 | 0.9252 | 0.9626 | 0.9589 |
| CTX | 0.7963 | 0.8889 | 0.5937 | 0.7954 | 0.8194 |
| CTZ | 0.7778 | 0.8340 | 0.5139 | 0.7564 | 0.7091 |
| GEN | 0.7716 | 0.7620 | 0.3930 | 0.6955 | 0.5789 |

CIP's XGBoost result closely matches the paper's reported XGBoost row (MCC 0.9253). GEN's XGBoost recall (0.5789) matches the paper's own XGBoost recall exactly — unlike the CNN, XGBoost did not collapse on the imbalanced antibiotic.

### Random Forest

Not yet reproduced. The original repo's Random Forest baseline (`Random Forest Implementations/AMR_Project_RF_Baseline_All.ipynb`) has been reviewed for provenance purposes but not run on this fork, since it is a baseline only and not part of the CNN+XGBoost ensemble this reproduction targets.

## Ensemble weight sweep (Stage 4 experiment)

The proposal's planned hyperparameter study: does the best CNN/XGBoost blend weight shift with class imbalance, and can a tuned weight beat the paper's fixed 50/50 average? Swept the blend weight from 0.0 (pure XGBoost) to 1.0 (pure CNN) in steps of 0.05, per antibiotic, choosing a threshold per weight via Youden's J on that weight's ROC curve.

| Antibiotic | 50/50 MCC | Best weight | Best MCC | MCC gain |
| :--- | :--- | :--- | :--- | :--- |
| CIP | 0.9008 | 0.00 (pure XGBoost) | 0.9252 | +0.0244 |
| CTX | 0.6706 | 0.65 (CNN-leaning) | 0.6750 | +0.0044 |
| CTZ | 0.5806 | 0.25 (XGBoost-leaning) | 0.5892 | +0.0087 |
| GEN | 0.4063 | 0.45 (~same as 0.5) | 0.4063 | 0.0000 |

Answer: no, 50/50 is not weight-optimal for three of the four antibiotics, though the gains are modest (+0.004 to +0.024 MCC). CIP shows the clearest case for moving off 50/50 toward pure XGBoost. The proposal's related ablation (standalone CNN vs. standalone XGBoost vs. full ensemble) is answered by the same data, since the per-model tables above plus the ensemble's 50/50 row together give all three numbers per antibiotic.

## Feature-selection k-sweep (second Stage 4 experiment, branch `M3_Own_Experiment`)

Branch: https://github.com/oliGAMER/ML_Paper_Reprod/tree/M3_Own_Experiment

Question: does XGBoost need the full 60,936 SNP features, or does a smaller, statistically-selected subset do just as well?

Method: chi-squared feature scoring (own implementation, categorical 5-way SNP tokens) on the training split only, per antibiotic; log-spaced k from 100 to 60,936 plus each antibiotic's Bonferroni-significant feature count; XGBoost retrained from scratch at each k.

| Antibiotic | Best k | Best MCC | Full-feature MCC | Bonferroni-significant features |
| :--- | :--- | :--- | :--- | :--- |
| CIP | ~3,500-3,900 | 0.9008 | 0.8770 (drops with more features) | 3,886 |
| CTX | 60,936 (full) | 0.6278 | 0.6278 | 786 |
| CTZ | ~415-847 | 0.5960 | 0.5806 | 724 |
| GEN | 415 | 0.4403 | 0.3930 (beats both full-feature and the paper's XGBoost MCC of 0.3722) | 349 |

Answer: yes, for CIP, CTZ and GEN, a few hundred to a few thousand well-chosen features (well under 10% of the full matrix) match or beat using everything. Only CTX clearly benefits from the full feature set. GEN's result is the standout: feature selection outperforms both this fork's full-feature XGBoost and the paper's own reported XGBoost MCC on the hardest, most imbalanced antibiotic. Full data and plots: `k_sweep_results.csv`, `k_sweep.png`, `chi2_significance_{CIP,CTX,CTZ,GEN}.png`.

Together, the weight sweep and the k-sweep answer both experiment slots committed to in the proposal (Section 7).

## Remaining work

- Class-weight formula: the original notebooks use standard inverse-frequency class weighting, but the paper's Section 3.4 states weights are inversely proportional to the square root of class frequency. Not yet resolved as a team whether to keep the code's actual formula (documenting the discrepancy) or implement the paper's stated formula as a deliberate deviation.
- GEN's CNN recall gap: try `val_recall` as the early-stopping monitor and a threshold re-sweep, since XGBoost's GEN recall is already much closer to the paper's CNN figure without any fix.
- Consolidate the intermediate `data_loader_full.py` into a single canonical `data_loader.py`.
- Sequence logos from the first CNN layer's learned filters, using the SNP CSVs, for the interpretability angle flagged in the deployment plan.
- Stage 5 deployment: a Streamlit app that takes one SNP sample and an antibiotic choice, and returns the predicted class, ensemble probability, and both component-model probabilities.
- Consider feeding the k-sweep's smaller feature sets into the deployed model, since they match or beat full-feature performance with a much smaller input.

Full detail, including bug fixes found in the original notebooks and the security incident writeup, is in `PROVENANCE.md`.
