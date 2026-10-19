"""
AMR-EnsembleNet demo app.

Streamlit UI for the reproduced 1D CNN + XGBoost soft-voting ensemble that
predicts antimicrobial resistance from SNP data. Pick an antibiotic and one
of the 162 held-out test isolates, and the app returns the CNN probability,
the XGBoost probability, the blended ensemble probability, and the predicted
label next to the real (held-out) ground truth.

Run with: streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.model_selection import train_test_split
from tensorflow import keras

REPO_ROOT = Path(__file__).resolve().parent.parent
SNP_PATH = REPO_ROOT / "Giessen_dataset" / "cip_ctx_ctz_gen_multi_data.csv"
PHENO_PATH = REPO_ROOT / "Giessen_dataset" / "cip_ctx_ctz_gen_pheno.csv"
RESULTS_DIR = REPO_ROOT / "results"

ANTIBIOTICS = ["CIP", "CTX", "CTZ", "GEN"]
FULL_NAMES = {
    "CIP": "Ciprofloxacin",
    "CTX": "Cefotaxime",
    "CTZ": "Ceftazidime",
    "GEN": "Gentamicin",
}
# Paper's fixed ensemble decision thresholds (Siddiqui & Tarannum, 2026),
# used for the fixed-threshold ensemble row in Section 6 of our report.
ENSEMBLE_THRESHOLDS = {"CIP": 0.55, "CTX": 0.48, "CTZ": 0.55, "GEN": 0.48}
CNN_WEIGHT = 0.5  # 50/50 soft vote, matching the paper's reported configuration
LABEL_NAME = {0: "Susceptible", 1: "Resistant"}


@st.cache_data(show_spinner="Loading SNP matrix and phenotype labels...")
def load_data():
    snp = pd.read_csv(SNP_PATH, index_col=0)
    pheno = pd.read_csv(PHENO_PATH, index_col=0)
    return snp, pheno


@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    cnn_models, xgb_models = {}, {}
    for ab in ANTIBIOTICS:
        cnn_models[ab] = keras.models.load_model(
            RESULTS_DIR / f"best_cnn1d_model_{ab}.keras"
        )
        clf = xgb.XGBClassifier()
        clf.load_model(RESULTS_DIR / f"best_xgboost_model_{ab}.json")
        xgb_models[ab] = clf
    return cnn_models, xgb_models


@st.cache_data(show_spinner=False)
def get_test_split(antibiotic: str):
    # Identical split logic to training/evaluation: an 80/20 stratified split,
    # seed 42, so this app only ever shows isolates the models never trained on.
    snp, pheno = load_data()
    X = snp.values.astype(np.int32)
    y = pheno[antibiotic].values
    ids = snp.index.to_numpy()
    _, X_test, _, y_test, _, id_test = train_test_split(
        X, y, ids, test_size=0.2, stratify=y, random_state=42
    )
    return X_test, y_test, id_test


def predict_one(antibiotic: str, row: np.ndarray):
    cnn_models, xgb_models = load_models()
    cnn_prob = float(cnn_models[antibiotic].predict(row[None, :], verbose=0).ravel()[0])
    xgb_prob = float(xgb_models[antibiotic].predict_proba(row[None, :])[0, 1])
    ens_prob = CNN_WEIGHT * cnn_prob + (1 - CNN_WEIGHT) * xgb_prob
    thr = ENSEMBLE_THRESHOLDS[antibiotic]
    pred = int(ens_prob >= thr)
    return cnn_prob, xgb_prob, ens_prob, pred, thr


st.set_page_config(page_title="AMR-EnsembleNet Demo", layout="centered")
st.title("AMR-EnsembleNet: Antimicrobial Resistance Predictor")
st.caption(
    "Reproduction of a 1D CNN + XGBoost soft-voting ensemble for predicting "
    "antibiotic resistance in *E. coli* from SNP data (Siddiqui & Tarannum, 2026)."
)

antibiotic = st.selectbox(
    "Antibiotic", ANTIBIOTICS, format_func=lambda a: f"{a} ({FULL_NAMES[a]})"
)

X_test, y_test, id_test = get_test_split(antibiotic)

isolate_id = st.selectbox(
    f"Test isolate ({len(id_test)} held out, never seen during training)",
    sorted(id_test.tolist()),
)

if st.button("Predict", type="primary"):
    idx = list(id_test).index(isolate_id)
    row = X_test[idx]
    true_label = int(y_test[idx])

    cnn_prob, xgb_prob, ens_prob, pred, thr = predict_one(antibiotic, row)

    col1, col2, col3 = st.columns(3)
    col1.metric("CNN probability (resistant)", f"{cnn_prob:.3f}")
    col2.metric("XGBoost probability (resistant)", f"{xgb_prob:.3f}")
    col3.metric("Ensemble probability (resistant)", f"{ens_prob:.3f}")

    st.divider()
    pred_col, true_col = st.columns(2)
    pred_col.subheader(f"Prediction: {LABEL_NAME[pred]}")
    pred_col.caption(f"Ensemble threshold for {antibiotic}: {thr}")
    true_col.subheader(f"Ground truth: {LABEL_NAME[true_label]}")
    if pred == true_label:
        true_col.success("Matches ground truth")
    else:
        true_col.error("Does not match ground truth")

with st.expander("About this demo"):
    st.markdown(
        """
- Models: a 1D CNN and an XGBoost classifier, trained separately per antibiotic
  on the GieSSen *E. coli* SNP dataset (Ren et al., 2022) — 60,936 SNP
  features, 809 isolates.
- The isolate list above is the real held-out 20% test split
  (`train_test_split`, `stratify`, `random_state=42`) used throughout our
  reproduction and report — these isolates were never used for training.
- Ensemble: a 50/50 soft vote of the CNN and XGBoost probabilities,
  thresholded at the paper's fixed per-antibiotic values (0.55 for CIP/CTZ,
  0.48 for CTX/GEN).
- See `ML_Paper_Reprod/report/M3_Report.tex` for full methodology and results.
        """
    )
