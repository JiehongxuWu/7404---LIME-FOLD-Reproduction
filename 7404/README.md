# LIME-FOLD Reproduction（AAAI-19）

Reproducing the paper:：**Induction of Non-Monotonic Logic Programs to Explain Boosted Tree Models Using LIME**（Farhad Shakerin, Gopal Gupta, AAAI-19）。

This repository implements **XGBoost → LIME（Algorithm 4）→ FOLD（Algorithm 3）** pipeline, also compared the baseline values ​​on multiple UCI datasets with those in **Table 1 / Figure 2** of the paper (ALEPH / ALEPH+LIME / FOLD+LIME are the values ​​in the paper's tables; **Your Result** is the reproduction of this project).

---

## Supported datasets

| Dataset | UCI ID | illustration |
|--------|--------|------|
| **heart** | 45 | Cleveland Heart Disease，Binary classification (whether or not one has the disease)|
| **wine** | 109 | Wine，The current implementation uses a binary classification system: **class 1 vs. non-class 1** (different from the three-class setting in the paper, see `REPRODUCTION_SETTINGS.md`). |
| **voting** | 105 | [Congressional Voting Records](https://archive.ics.uci.edu/dataset/105/congressional+voting+records)，Binary classification（Republican vs Democrat） |

---

## Project Structure (Brief)

```
7404/
├── data/processed/          # Preprocess CSV (git does not commit by default)
├── results/
│   ├── models/              # Each dataset includes XGBoost, LIME pkl, and feature_meta.json.
│   ├── figures/             # Comparison charts and summary tables
│   └── logs/                # CSV Metrics
├── src/
│   ├── data_processing/     # heart / wine / voting preprocess，MDLP，public preprocess_utils
│   ├── explanation/         # LIME
│   └── models/              # FOLD + LIME-FOLD main process
├── experiments/
│   ├── run_heart_experiment.py   # One-click: Step 1 → Lime → FOLD → Output Image
│   └── generate_table1_ours.py   # Table 1: Style Table (summarizing three datasets)
├── REPRODUCTION_SETTINGS.md      # Reproduction setup (data, metrics, environment, implementation details)
└── requirements.txt
```

---

## Environment

- **Python**：Version 3.8 is recommended (validate under 3.8.0 during development).
- **Virtual environment**（recommend）：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running mode

Execute after the project root directory and `.venv` is activated.

### 1）For each dataset Step1：Preprocessing + Training XGBoost

```bash
python -m src.data_processing.heart_preprocessing
python -m src.data_processing.wine_preprocessing
python -m src.data_processing.voting_preprocessing
```

Product examples (named by dataset prefix)：

- `data/processed/{dataset}_X_final.csv`、`{dataset}_y_binary.csv`
- `results/models/{dataset}_xgb_model.json`、`{dataset}_feature_meta.json`

> Default **NOT** The XGBoost feature importance figure in Figure 3 of the paper was generated.。

### 2）Complete Experiment: LIME → LIME-FOLD → 5-fold index → ​​Comparison chart

```bash
python experiments/run_heart_experiment.py --dataset heart
python experiments/run_heart_experiment.py --dataset wine
python experiments/run_heart_experiment.py --dataset voting
```

Optional parameter: `--num-features K` (LIME interpretation length, default 5).

OUTPUT：

- `results/logs/{dataset}_lime_fold_results.csv`
- `results/figures/{dataset}_experiment_comparison.png`(Left: F1; Right: Rule number, **two decimal places**)

### 3）Summary table of three datasets (our own values)

```bash
python experiments/generate_table1_ours.py
```

Generate：`results/logs/table1_ours.csv`、`results/figures/table1_ours.png`。

---

## Paper and Reproduction Instructions

- The **ALEPH / ALEPH+LIME / FOLD+LIME** in the figure are from the paper **Table 1 / Figure 2**, and are not running Aleph on this computer.
- Numerical values ​​are not guaranteed to be completely consistent with those in the paper：See root directory **`REPRODUCTION_SETTINGS.md`**。

---

## Cite

If you use this repository, please also cite the original paper and the UCI dataset DOI (see each dataset page).
