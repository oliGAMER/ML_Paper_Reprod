"""
AMR-EnsembleNet demo app.

Streamlit UI for the reproduced models that predict antimicrobial resistance
from SNP data. Pick an antibiotic, a feature set, and one of the 162
held-out test isolates:

- Full (60,936 features): the 1D CNN + XGBoost soft-voting ensemble from
  Sections 4 and 6 of the report.
- Optimized (chi-squared selected): the XGBoost-only model from the
  feature-selection k-sweep (Section 7.2), trained on a much smaller,
  statistically chosen subset of features. No CNN is used in this mode.

Run with: streamlit run app.py
"""

import json
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
LABEL_NAME = {0: "Susceptible", 1: "Resistant"}

# --- Full-feature ensemble (Sections 4 and 6) ---
# Paper's fixed ensemble decision thresholds (Siddiqui & Tarannum, 2026).
ENSEMBLE_THRESHOLDS = {"CIP": 0.55, "CTX": 0.48, "CTZ": 0.55, "GEN": 0.48}
CNN_WEIGHT = 0.5  # 50/50 soft vote, matching the paper's reported configuration

# --- Optimized-feature, XGBoost-only model (Section 7.2 k-sweep) ---
# Best-MCC k per antibiotic from m3/k_sweep_results.csv (smallest k on ties).
OPTIMIZED_K = {"CIP": 3524, "CTX": 60936, "CTZ": 415, "GEN": 415}
# Same thresholds the k-sweep notebook itself used (reused from the CNN's
# per-antibiotic early-stopping thresholds; not the ensemble thresholds above).
OPTIMIZED_THRESHOLDS = {"CIP": 0.30, "CTX": 0.50, "CTZ": 0.62, "GEN": 0.505}
TOTAL_FEATURES = 60936


@st.cache_data(show_spinner="Loading SNP matrix and phenotype labels...")
def load_data():
    snp = pd.read_csv(SNP_PATH, index_col=0)
    pheno = pd.read_csv(PHENO_PATH, index_col=0)
    return snp, pheno


@st.cache_resource(show_spinner="Loading full-feature models...")
def load_full_models():
    cnn_models, xgb_models = {}, {}
    for ab in ANTIBIOTICS:
        cnn_models[ab] = keras.models.load_model(
            RESULTS_DIR / f"best_cnn1d_model_{ab}.keras"
        )
        clf = xgb.XGBClassifier()
        clf.load_model(RESULTS_DIR / f"best_xgboost_model_{ab}.json")
        xgb_models[ab] = clf
    return cnn_models, xgb_models


@st.cache_resource(show_spinner="Loading optimized-feature models...")
def load_optimized_models():
    xgb_models, feature_indices = {}, {}
    for ab in ANTIBIOTICS:
        clf = xgb.XGBClassifier()
        clf.load_model(RESULTS_DIR / f"best_xgboost_optimized_{ab}.json")
        xgb_models[ab] = clf
        with open(RESULTS_DIR / f"optimized_feature_indices_{ab}.json") as f:
            feature_indices[ab] = np.array(json.load(f)["selected_column_indices"])
    return xgb_models, feature_indices


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


def predict_full(antibiotic: str, row: np.ndarray):
    cnn_models, xgb_models = load_full_models()
    cnn_prob = float(cnn_models[antibiotic].predict(row[None, :], verbose=0).ravel()[0])
    xgb_prob = float(xgb_models[antibiotic].predict_proba(row[None, :])[0, 1])
    ens_prob = CNN_WEIGHT * cnn_prob + (1 - CNN_WEIGHT) * xgb_prob
    thr = ENSEMBLE_THRESHOLDS[antibiotic]
    pred = int(ens_prob >= thr)
    return cnn_prob, xgb_prob, ens_prob, pred, thr


def predict_optimized(antibiotic: str, row: np.ndarray):
    xgb_models, feature_indices = load_optimized_models()
    reduced_row = row[feature_indices[antibiotic]]
    xgb_prob = float(xgb_models[antibiotic].predict_proba(reduced_row[None, :])[0, 1])
    thr = OPTIMIZED_THRESHOLDS[antibiotic]
    pred = int(xgb_prob > thr)  # matches the k-sweep notebook's own ">" comparison
    return xgb_prob, pred, thr


st.set_page_config(page_title="AMR-EnsembleNet Demo", layout="centered")
st.title("AMR-EnsembleNet: Antimicrobial Resistance Predictor")
st.caption(
    "Reproduction of a 1D CNN + XGBoost soft-voting ensemble for predicting "
    "antibiotic resistance in *E. coli* from SNP data (Siddiqui & Tarannum, 2026)."
)

antibiotic = st.selectbox(
    "Antibiotic", ANTIBIOTICS, format_func=lambda a: f"{a} ({FULL_NAMES[a]})"
)

feature_set = st.radio(
    "Feature set",
    ["Full (60,936 features)", "Optimized (chi-squared selected)"],
    captions=[
        "1D CNN + XGBoost ensemble, Sections 4 and 6",
        "XGBoost only, no CNN — Section 7.2 k-sweep",
    ],
)

k = OPTIMIZED_K[antibiotic]
if feature_set.startswith("Optimized") and k == TOTAL_FEATURES:
    st.info(
        f"For {antibiotic}, the optimized feature count equals the full set "
        f"({k:,} of {TOTAL_FEATURES:,}) — this is the one antibiotic where "
        "the k-sweep found no smaller subset that matched full-set performance."
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

    if feature_set.startswith("Full"):
        cnn_prob, xgb_prob, ens_prob, pred, thr = predict_full(antibiotic, row)
        col1, col2, col3 = st.columns(3)
        col1.metric("CNN probability (resistant)", f"{cnn_prob:.3f}")
        col2.metric("XGBoost probability (resistant)", f"{xgb_prob:.3f}")
        col3.metric("Ensemble probability (resistant)", f"{ens_prob:.3f}")
        threshold_caption = f"Ensemble threshold for {antibiotic}: {thr}"
    else:
        xgb_prob, pred, thr = predict_optimized(antibiotic, row)
        col1, col2 = st.columns(2)
        col1.metric("XGBoost probability (resistant)", f"{xgb_prob:.3f}")
        col2.metric("Features used", f"{k:,} / {TOTAL_FEATURES:,} ({k / TOTAL_FEATURES:.1%})")
        threshold_caption = f"Decision threshold for {antibiotic}: {thr}"

    st.divider()
    pred_col, true_col = st.columns(2)
    pred_col.subheader(f"Prediction: {LABEL_NAME[pred]}")
    pred_col.caption(threshold_caption)
    true_col.subheader(f"Ground truth: {LABEL_NAME[true_label]}")
    if pred == true_label:
        true_col.success("Matches ground truth")
    else:
        true_col.error("Does not match ground truth")

with st.expander("About this demo"):
    st.markdown(
        """
- Dataset: the GieSSen *E. coli* SNP dataset (Ren et al., 2022) — 60,936 SNP
  features, 809 isolates.
- The isolate list above is the real held-out 20% test split
  (`train_test_split`, `stratify`, `random_state=42`) used throughout our
  reproduction and report — these isolates were never used for training.
- **Full** mode: a 1D CNN and an XGBoost classifier, each trained separately
  per antibiotic on all 60,936 features, blended 50/50 and thresholded at
  the paper's fixed per-antibiotic values (0.55 for CIP/CTZ, 0.48 for
  CTX/GEN).
- **Optimized** mode: an XGBoost-only model trained on a much smaller,
  chi-squared-selected feature subset (Section 7.2's k-sweep). No CNN is
  used. Thresholds here (0.30/0.50/0.62/0.505) match the k-sweep notebook's
  own evaluation, not the ensemble thresholds above.
- See `ML_Paper_Reprod/report/M3_Report.tex` for full methodology and results.
        """
    )
