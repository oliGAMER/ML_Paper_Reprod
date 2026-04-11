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
