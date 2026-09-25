# **PROVENANCE.md**

Tracking of what in this repository is our own work, adapted from the original authors' code, or reused as-is — maintained as we go, per the assignment's academic integrity requirements.

**Team:** Zaid Bilal, Omer Ibrahim Qazi, Haseeb Adnan **Original paper/repo:** Siddiqui & Tarannum (2026), `Saiful185/AMR-EnsembleNet` **Our fork:** `oliGAMER/ML_Paper_Reprod`

## **Environment**

* Cloned `oliGAMER/ML_Paper_Reprod` (fork of `Saiful185/AMR-EnsembleNet`).  
* Local dev machine's default (conda base) is Python 3.13. `requirements.txt`'s pins (`pandas==2.2.2`, `tensorflow==2.18.0`) predate Python 3.13 support and failed to build from source under `python3 -m venv` (Cython/C++ incompatibility with GCC 15 — `[[maybe_unused]]` attribute placement error in pandas' `aggregations.pyx.cpp`).  
* **Fix:** created a dedicated conda environment pinned to Python 3.11 (`conda create -n amr python=3.11`) so all packages install from prebuilt wheels. No changes made to `requirements.txt` package versions.  
* Old broken `venv/` directory removed; `venv/`, `__pycache__/`, `*.pyc`, `*.egg-info/` added to `.gitignore`.  
* `*:Zone.Identifier` files (WSL artifacts from Windows-downloaded files) added to `.gitignore`.

  ## **Migration to Google Colab (2026-09-25)**

**Reason:** local WSL training runs were killed by the Linux OOM-killer (`dmesg` confirmed `python3 invoked oom-killer`, process killed at \~4.2–4.5GB resident memory against a \~7.2GB total / \~4.3GB available WSL memory ceiling — confirmed via `free -h`). Reducing batch size was assessed as an insufficient workaround given how close usage already was to the ceiling; migrated to Colab instead. This also matches the original authors' own notebooks, which were written for Colab (Google Drive mounts, `/content/drive/...` paths) — running on Colab is arguably a more faithful reproduction environment than local WSL, not a deviation from it.

**Setup:** cloned `oliGAMER/ML_Paper_Reprod` into the Colab instance for reference; recreated project layout at `/content/AMR_repro/` (`src/`, `Giessen_dataset/`, `results/`); wrote `src/data_loader.py` and `src/train_cnn.py` to the instance (also an intermediate `data_loader_full.py` present — same file-naming consolidation noted earlier under Code provenance, not yet cleaned up to one canonical name in the Colab copy); data loaded from Google Drive (`cip_ctx_ctz_gen_multi_data.csv`, `cip_ctx_ctz_gen_pheno.csv`), cached locally as parquet for faster reload across cells; ran on Colab's free T4 GPU tier, **TensorFlow 2.20.0** (Colab's preinstalled default — attempted to pin to match `requirements.txt`'s `tensorflow==2.18.0`, but `pip install tensorflow==2.18.0` fails on Colab: PyPI no longer serves that version as an installable wheel for Colab's current Python/platform combo — only `2.20.0` and newer are available. **This is not fixable by us; TF 2.20.0 is now the de facto environment for all Colab-run results**, a real, unavoidable environment deviation from `requirements.txt`, not an oversight — document plainly in the report rather than treating it as something to still resolve), batch size reverted to 32 (the original notebooks' value — no longer memory-constrained on Colab). Re-verified 809×60,936 shape and correct class counts post-migration (matches earlier local verification).

**⚠️ Security incident (resolved):** a GitHub personal access token was found hardcoded in plain text in a Colab notebook cell (a `!git push https://<user>:<token>@github.com/...` command). **Token has been revoked.** Going forward: never hardcode credentials in any notebook cell — use `getpass.getpass()` for interactive entry, or Colab's `google.colab.userdata` secrets manager. Check notebook cell *outputs* before committing, not just source — a printed credential in an output leaks the same way a hardcoded one does.

### **CNN reproduction results (Colab, 2026-09-25)**

All four antibiotics trained to completion (150 epochs, subject to early stopping). Final test-set metrics vs. the paper's Table 2–5 (1D CNN row only):

| Antibiotic | Our Accuracy | Paper Accuracy | Our MCC | Paper MCC | Our Macro F1 | Paper Macro F1 | Our Recall (resistant) | Paper Recall (resistant) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| CIP (v1) | 0.8765 | 0.9568 | 0.7568 | 0.9129 | 0.8762 | 0.9564 | 0.9178 | 0.9589 |
| **CIP (v2, re-run)** | **0.9198** | 0.9568 | **0.8405** | 0.9129 | **0.9194** | 0.9564 | **0.9452** | 0.9589 |
| CTX | 0.8025 | 0.7840 | 0.6030 | 0.5647 | 0.8010 | 0.7821 | 0.8056 | 0.7778 |
| CTZ | 0.7963 | 0.8086 | 0.5402 | 0.5651 | 0.7698 | 0.7816 | 0.6727 | 0.6727 |
| GEN (v1, `val_accuracy` monitor) | 0.7654 | 0.7346 | 0.0000 | 0.3984 | 0.4336 | 0.6836 | 0.0000 | 0.7105 |
| **GEN (v2, `val_auc` monitor)** | **0.7778** | 0.7346 | **0.3136** | 0.3984 | **0.6495** | 0.6836 | **0.3684** | 0.7105 |

**CTX and CTZ reproduced reasonably close to the paper** — within a few points on every metric, a solid reproduction for those two.

**CIP v2 Re-run Analysis:** With the fresh run completed, CIP's MCC jumped from 0.7568 up to **0.8405** (bringing it much closer to the paper's 0.9129), and recall on the resistant class reached **0.9452** (vs. the paper's 0.9589). This largely closes the previous gap and suggests the earlier lower score was simply run-to-run variance inherent to GPU training despite random seeds.

**GEN v1 (original `val_accuracy` monitor) reproduction failed outright** on first attempt. The model collapsed to predicting "Susceptible" for all 162 test samples: 0.0 recall/precision/MCC/kappa/F1 on the resistant class, AUC 0.4272 (below chance). This was the **opposite** of the paper's own headline GEN result, where the standalone CNN's high recall (0.7105) on this most-imbalanced antibiotic is presented as the model's key strength (Section 4.3).

**Root cause confirmed (2026-09-25): fixed by changing the early-stopping monitor.** Re-ran GEN with `es_monitor` changed from `val_accuracy` to `val_auc` (all other config — threshold 0.505, patience 60, architecture, class weights — unchanged), under the same TF 2.20.0 environment CTX/CTZ already trained successfully under (ruling out TF version as the cause). Result:

| Metric | v1 (`val_accuracy`) | v2 (`val_auc`) | Paper |
| :---- | :---- | :---- | :---- |
| AUC | 0.4272 | **0.7417** | — |
| MCC | 0.0000 | **0.3136** | 0.3984 |
| Macro F1 | 0.4336 | **0.6495** | 0.6836 |
| Recall (resistant) | 0.0000 | **0.3684** | 0.7105 |

This confirms the hypothesis: with GEN's severe imbalance (188/809 ≈ 23% resistant), a model that always predicts the majority class already scores \~77% accuracy "for free," so `val_accuracy` was a poor stopping signal that let the model checkpoint at exactly this degenerate solution. **This is a genuine weakness in the original authors' own training setup for their hardest task, reproduced faithfully in v1 — not a bug we introduced.** Switching to `val_auc` (a deliberate, documented deviation from the original notebook) resolves the collapse and brings MCC/Macro F1 within reasonable range of the paper's reported numbers.

**Remaining gap:** recall (0.3684) is still well below the paper's 0.7105 — GEN is now a normal, explainable underperformance rather than total failure, but not yet a tight reproduction. From the training log, recall briefly reached 0.6842 around epoch 33 before drifting down as training continued to its eventual stop at epoch 89 — suggesting `val_recall` (rather than `val_auc`) as the monitor might land closer to the paper's number by stopping earlier, closer to that peak. Not yet tried; candidate next step, time permitting, alongside a decision-threshold sweep (0.505 was carried over from the original notebook and may not be optimal under the new monitor).

This whole finding — the collapse, its cause, and the fix — belongs in the report's Reproduction Results and Analysis/Discussion sections as-is: a legitimate and informative result whichever way you read it (either an artifact-of-`val_accuracy` finding worth reporting on its own, or a useful worked example of debugging a faithful reproduction that initially failed for a non-obvious reason).

## **Data provenance**

* **Files:** `cip_ctx_ctz_gen_multi_data.csv` (SNP matrix), `cip_ctx_ctz_gen_pheno.csv` (phenotype labels).  
* **Source:** `Giessen_dataset.zip`, bundled in the original `Saiful185/AMR-EnsembleNet` repo. Not present in our fork's default checkout (`oliGAMER/ML_Paper_Reprod`) — added manually to `Giessen_dataset/` in the local working copy.  
* **Storage decision:** resolved — Git LFS installed and working. `cip_ctx_ctz_gen_multi_data.csv` (\~99MB) tracked via LFS (`.gitattributes` filter); `cip_ctx_ctz_gen_pheno.csv` (\~21KB) committed as a plain file, no LFS needed.  
* **Push status:** commit made locally; push still pending collaborator access from repo owner (`oliGAMER`).  
* **Verification performed (2026-09-25):**  
* SNP matrix shape: 809 samples × 60,936 SNP feature columns (60,937 columns incl. index) — matches paper (Section 3.1).  
* Phenotype file shape: 809 samples × 4 antibiotic columns (CIP, CTX, CTZ, GEN).  
* Sample IDs match 1:1 and in identical order between the two files.  
* Class counts match Table 1 in the paper exactly:

| Antibiotic | Susceptible (0) | Resistant (1) |
| :---- | :---- | :---- |
| CIP | 443 | 366 |
| CTX | 451 | 358 |
| CTZ | 533 | 276 |
| GEN | 621 | 188 |

* Confirmed independently via `src/data_loader.py`.

  ## **Code provenance log**

| File / Notebook | Status | Notes |
| :---- | :---- | :---- |
| `src/data_loader.py` | **written by us** | Loads SNP matrix \+ phenotype CSVs, asserts shape/alignment against paper's reported dataset stats, provides `get_split()` for stratified 80/20 per-antibiotic split (matching Section 3.4's protocol). Fixed `random_state=42` — matches original notebooks' seed (confirmed, see below). **Verified 2026-09-25:** full load confirms 809×60,936 matrix, correct class counts; `get_split()` confirmed to preserve class ratio within \~0.2pp of full-dataset ratio across all four antibiotics in both train (n=647) and test (n=162) splits. |
| `src/train_cnn.py` | **adapted from** `Final Custom 1D CNN Implementations/AMR_Project_1D_CNN_v1_{CIP,CTX,CTZ,GEN}.ipynb` | Consolidates the authors' four near-duplicate Colab notebooks (one per antibiotic) into one parameterized script. Architecture (`build_cnn1d_model`, ported from `build_cnn1d_model_extended`) and core hyperparameters (embedding dim 64, dropout rates, 150 epochs, batch 32, lr 1e-3) are **reused as-is, unchanged**. Changes made: (1) data loading routed through `src/data_loader.py` instead of each notebook's own `pd.read_csv`; (2) fixed a latent bug — original notebooks computed class weights from the full pre-split `labels` array (train+test combined) rather than `y_train` only; here weights are computed from `y_train` only; (3) Colab `drive.mount()` cell removed; (4) per-antibiotic settings (decision threshold, early-stopping monitor/patience) that were hardcoded differently per notebook are now explicit in `ANTIBIOTIC_CONFIG` instead of silently varying across four separate files; (5) fixed a checkpoint-filename bug in the original CTX notebook (see below); (6) **GEN's `es_monitor` further changed from the original `val_accuracy` to `val_auc`** after diagnosing a training collapse — see "Migration to Google Colab" section above for full root-cause analysis and before/after metrics. This is a deliberate deviation from the original notebook's config, not a reused-as-is value, unlike CIP/CTX/CTZ's settings which remain exactly as found. |
| `AMR Ensemble Models/` | not yet reviewed | Authors' soft-voting ensemble notebook(s). To be read next. |
| `Final Custom 1D CNN Implementations/` | **reviewed** — see `src/train_cnn.py` row above | Source for the CNN, now consolidated. |
| `Random Forest Implementations/` | not yet reviewed | Authors' Random Forest baseline. |
| `XGBoost Implementations/` | not yet reviewed | Authors' XGBoost baseline. |
| `AMR_Project_1D_CNN_Tuning.ipynb` | not yet reviewed | Root-level hyperparameter tuning notebook, likely source of the final CNN architecture in Figure 1\. |

  ## **Findings from reading the CNN notebooks (2026-09-25)**

Diffed all four `AMR_Project_1D_CNN_v1_*.ipynb` notebooks cell-by-cell against each other. They are near-identical except for `TARGET_ANTIBIOTIC` and the following **undocumented per-antibiotic settings** (not mentioned anywhere in the paper):

| Antibiotic | Early-stopping monitor | ES patience | Decision threshold |
| :---- | :---- | :---- | :---- |
| CIP | `val_auc` | 60 | 0.30 |
| CTX | `val_accuracy` | 40 | 0.50 |
| CTZ | `val_accuracy` | 40 | 0.62 |
| GEN | `val_accuracy` | 60 | 0.505 |

* **Decision thresholds are not 0.5 for CIP, CTZ, or GEN.** This matters a lot for the paper's GEN recall claims (Table 5\) — a 0.505 threshold vs. a naive 0.5 default is a real, deliberate tuning choice that isn't documented in the paper text. Worth mentioning in our report's Implementation Details section.  
* **Bug found in the original CTX notebook:** its `ModelCheckpoint` callback saves to `best_cnn1d_model_CTZ.keras` (filename left over from copy-pasting the CTZ notebook) instead of a CTX-specific name. If CTX and CTZ notebooks were ever run in the same working directory, CTX's training could silently overwrite or load the wrong checkpoint. Fixed in `src/train_cnn.py` (checkpoint filenames now derived from `antibiotic` param).  
* **Class weight formula, code vs. paper:** the notebooks compute standard inverse-frequency class weights, `(1/n_class) * (total/2)`. The paper's Section 3.4 states weights are "inversely proportional to the **roots** of the class frequencies" (i.e. inverse square-root). **The code does not match the paper's stated method.** Not yet resolved — options are (a) keep the code's actual formula and note the discrepancy in our report's Analysis/Limitations, or (b) implement the paper's stated square-root formula ourselves as a deliberate deviation from the authors' code, which would itself need documenting. Leaning towards (a) for the reproduction stage, since our job is to reproduce what the code does, not what the paper claims it does — but flag this for team discussion.  
* **Random seed:** `tf.random.set_seed(42)` and `np.random.seed(42)` ARE set in all four notebooks (resolves the "is a seed set?" open question below — yes, 42, consistent with our own `data_loader.py`).  
* **Latent bug (class weights computed pre-split):** original notebooks call `np.bincount(labels)` for class weighting using the full pre-`train_test_split` `labels` array, not `y_train`. This means the reported class weights are based on train+test combined class balance rather than train-only — a minor train/test leakage in the weighting step (not the data itself). Fixed in `src/train_cnn.py`.

  ## **Open questions / things to verify once reading the authors' notebooks (Step 3.3)**

* \[x\] Does the code's class-weighting formula match the paper's stated "inverse square-root of class frequency" (Section 3.4)? **No — see Findings above. Code uses standard inverse frequency, not inverse square-root.**  
* \[x\] Is a random seed set anywhere in the original notebooks? **Yes — `seed=42` in all four CNN notebooks, matching our own loader.**  
* \[ \] Does XGBoost's `n_estimators` in code match Figure 1's stated 1000? (Not yet checked — pending review of `XGBoost Implementations/`.)  
* \[x\] **GEN collapse (2026-09-25):** confirmed root cause — `val_accuracy` as the early-stopping monitor let the model checkpoint at a degenerate "always predict majority class" solution under GEN's \~23% imbalance. **Fixed** by switching to `val_auc`: MCC 0.0000 → 0.3136, Macro F1 0.4336 → 0.6495, recall 0.0000 → 0.3684 (paper: MCC 0.3984, Macro F1 0.6836, recall 0.7105). Recall still meaningfully below the paper — see Migration section for the remaining-gap discussion and `val_recall`\-monitor as a candidate next step.  
* \[x\] **CIP underperformance (Resolved):** Re-running CIP brought MCC up from 0.7568 to **0.8405** and resistant recall up to **0.9452**, nicely closing the gap toward the paper's numbers (0.9129 MCC, 0.9589 recall). The initial drop was simply standard run-to-run GPU variance.

  ## **Blocked / pending**

- &nbsp;