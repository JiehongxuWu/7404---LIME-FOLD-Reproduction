# Reproduction Settings

This document summarizes the **data, evaluation metrics, operating environment, key implementation and training details** of this repository when reproducing the AAAI-19 paper *LIME-FOLD*, to facilitate report writing and review.

---

## 1. Data

### 1.1 Heart Disease（Cleveland）

- **Source**：UCI，`ucimlrepo.fetch_ucirepo(id=45)`  
- **Task**: Binary classification, label `y_binary = (num > 0)` (whether or not the disease is present). 
- **Feature**：Numerical columns are **discretized using MDLP** and then used as an interval with the first 4 columns; categorical columns are one-hot encoded with semantic column names (e.g., `chest_pain_4`, `thal_7`).
- **Object prefix**：`heart_*`（如 `heart_X_final.csv`, `heart_xgb_model.json`）。

### 1.2 Wine

- **Source**：UCI id=109，[Wine](https://archive.ics.uci.edu/dataset/109/wine)。  
- **Taske**：The original data was in **3 categories**; this repository implements it as **binary classification**: **whether it belongs to class 1** (`class == 1` is a positive class).
  - This is inconsistent with the **multi-class** specification of the dataset in the paper, therefore the **F1 score** is incorrect.
- **Object prefix**：`wine_*`.

### 1.3 Congressional Voting Records

- **Source**：UCI id=105，[Congressional Voting Records](https://archive.ics.uci.edu/dataset/105/congressional+voting+records).
- **Task**：Binary classification, **Republican = 1**, **Democrat = 0**.
- **Lose**：In the voting column, `?` is mapped to the string `missing` and then participates in one-hot encoding.  
- **Object prefix**：`voting_*`.

### 1.4 Data storage location

- Preprocessing：`data/processed/{dataset}_X_final.csv`, `{dataset}_y_binary.csv`  
- Meta information：`results/models/{dataset}_feature_meta.json`（include `interval_features`、`target_predicate`）

---

## 2. Evaluation Metrics

Consistent with the experimental section of the paper, the rules learned by **LIME-FOLD** are calculated on the test fold:

| Metrics | Description |
|------|------|
| **Precision** | Precision for the positive class (predicted as 1) |
| **Recall** | Recall for the positive class |
| **Accuracy** | Accuracy |
| **F1** | F1 score for the positive class |
| **Avg. number of rules** | he sum of the number of **default + abnormal** rules per 5-fold division, then averaged|

**Division method**：`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`。

**Warning**：The values ​​for **ALEPH / ALEPH+LIME / FOLD+LIME** in the figure are from **Table 1 / Figure 2 of the paper**, and are not from running Aleph on the local computer; **Your Result** is the pipeline output of this project.

---

## 3. Environment

| 项目 | 说明 |
|------|------|
| **OS** | Windows 10(Also runs on Linux/macOS)|
| **Python** | Recommended **3.8+** (Repository used under 3.8.0)|
| **Virtual Environment** | Recommended project root directory `.venv/`|
| **Dependencies** | See `requirements.txt`（pandas, numpy, xgboost, scikit-learn, lime, matplotlib, seaborn, ucimlrepo） |

Install:

```bash
pip install -r requirements.txt
```

---

## 4. key Implementation & Training Deatils

### 4.1 XGBoost（Step 1）

- Each datasets share the same parameters in `preprocess_utils._build_xgb()`(`n_estimators=600`, `max_depth=15`, `learning_rate=0.02` etc).
- Fit once on the **complete training set** (unlike the "retraining black box per fold" implementation in some papers, if strict alignment is required, it can be changed to retraining within the CV).

### 4.2 Numerical characteristics: MDLP discretization

- Implementation：`src/data_processing/mdlp_discretizer.py`(Fayyad & Irani style MDLP).
- Typical parameters：`min_depth=1`, `max_depth=6`, `min_samples=10`.  
- For datasets with **no numerical columns** (such as fully discrete voting datasets): `preprocess_utils` performs only one-hot encoding and **does not perform MDLP**.

### 4.3 LIME（Algorithm 4 First Half）

- `lime.lime_tabular.LimeTabularExplainer`，`discretize_continuous=False`，Features are processed by categorical.  
- Default values ​​are `num_features=5` and `num_samples=5000` (which can be adjusted using `--num-features` in `run_heart_experiment.py`).
- Interpretation writing `results/models/{dataset}_lime_explanations.pkl`.

### 4.4 Dataset Transformation + FOLD

- **E+/E−**：Constructed from **model-predicted labels** `M(r)` (not just the true labels). 
- **BK**：The literal is generated from LIME’s `feature_weights_by_class[1]` (positive class) and the sample feature values; **positive/negative weights are retained** (not discarded according to their signs).
- **interval / binary**：The value is determined by the `interval_features` parameter in `feature_meta.json` and the column values. 
- **FOLD**：`max_rule_length=5`，The target predicates are available in each dataset `target_predicate`（如 `heart_disease`, `wine_target`, `voting_target`）。

### 4.5 Figure

- **NOT generate** Figure 3 in the paper shows the global feature importance map for each class in XGBoost (currently disabled by default).。  
- Comparison Figures：`results/figures/{dataset}_experiment_comparison.png`，The column number is labeled with **two decimal places** for easy comparison with the decimal column numbers in Figure 2 of the paper.

---

## 5. Common reasons for discrepancies between numerical values ​​and those in the paper

- The XGBoost/LIME/FOLD implementations are not entirely consistent with the original author's Java+Prolog stack；  
- **Binary classification of wine vs. multi-class classification of papers** This means that indicators and the number of rules cannot be directly equated.；  
- 5-fold Random seed and single partition fluctuation；  
- MDLP and engineering parameters such as pruning threshold.

For a more complete discussion, please see the project's experimental log and `results/logs/*.csv`。
