# LIME-FOLD 复现（AAAI-19）

复现论文：**Induction of Non-Monotonic Logic Programs to Explain Boosted Tree Models Using LIME**（Farhad Shakerin, Gopal Gupta, AAAI-19）。

本仓库实现 **XGBoost → LIME（Algorithm 4）→ FOLD（Algorithm 3）** 流水线，并在多个 UCI 数据集上与论文 **Table 1 / Figure 2** 中的基线数值作对照（ALEPH / ALEPH+LIME / FOLD+LIME 为论文表内数值；**Your Result** 为本项目复现）。

---

## 支持的数据集

| 数据集 | UCI ID | 说明 |
|--------|--------|------|
| **heart** | 45 | Cleveland Heart Disease，二分类（患病与否） |
| **wine** | 109 | Wine，当前实现为 **class 1 vs 非 class 1** 二分类（与论文 3 类设定不同，见 `REPRODUCTION_SETTINGS.md`） |
| **voting** | 105 | [Congressional Voting Records](https://archive.ics.uci.edu/dataset/105/congressional+voting+records)，二分类（Republican vs Democrat） |

---

## 项目结构（简要）

```
7404/
├── data/processed/          # 预处理 CSV（默认不提交 git）
├── results/
│   ├── models/              # 各数据集 XGBoost、LIME pkl、feature_meta.json
│   ├── figures/             # 对比图、汇总表图
│   └── logs/                # CSV 指标
├── src/
│   ├── data_processing/     # heart / wine / voting 预处理，MDLP，公共 preprocess_utils
│   ├── explanation/         # LIME
│   └── models/              # FOLD + LIME-FOLD 主流程
├── experiments/
│   ├── run_heart_experiment.py   # 一键：Step1→LIME→FOLD→出图
│   └── generate_table1_ours.py   # 汇总三数据集 Table1 风格表
├── REPRODUCTION_SETTINGS.md      # 复现设置（数据、指标、环境、实现细节）
└── requirements.txt
```

---

## 环境

- **Python**：建议 3.8+（开发时在 3.8.0 下验证）
- **虚拟环境**（推荐）：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 运行方式

项目根目录、已激活 `.venv` 后执行。

### 1）各数据集 Step1：预处理 + 训练 XGBoost

```bash
python -m src.data_processing.heart_preprocessing
python -m src.data_processing.wine_preprocessing
python -m src.data_processing.voting_preprocessing
```

产物示例（按数据集前缀命名）：

- `data/processed/{dataset}_X_final.csv`、`{dataset}_y_binary.csv`
- `results/models/{dataset}_xgb_model.json`、`{dataset}_feature_meta.json`

> 默认 **不** 生成论文 Figure 3 的 XGBoost 特征重要性图。

### 2）完整实验：LIME → LIME-FOLD → 5-fold 指标 → 对比图

```bash
python experiments/run_heart_experiment.py --dataset heart
python experiments/run_heart_experiment.py --dataset wine
python experiments/run_heart_experiment.py --dataset voting
```

可选参数：`--num-features K`（LIME 解释长度，默认 5）。

输出：

- `results/logs/{dataset}_lime_fold_results.csv`
- `results/figures/{dataset}_experiment_comparison.png`（左：F1；右：规则数，**小数两位**标注）

### 3）三数据集汇总表（我们自己的数值）

```bash
python experiments/generate_table1_ours.py
```

生成：`results/logs/table1_ours.csv`、`results/figures/table1_ours.png`。

---

## 论文与复现说明

- 图中 **ALEPH / ALEPH+LIME / FOLD+LIME** 来自论文 **Table 1 / Figure 2**，非本机运行 Aleph。
- 数值与论文完全一致不保证：见根目录 **`REPRODUCTION_SETTINGS.md`**。

---

## 引用

若使用本仓库，请同时引用原论文与 UCI 数据集 DOI（见各数据集页面）。
