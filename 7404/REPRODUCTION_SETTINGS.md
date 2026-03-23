# 复现设置（Reproduction Settings）

本文档归纳本仓库在复现 AAAI-19 论文 *LIME-FOLD* 时的**数据、评估指标、运行环境、关键实现与训练细节**，便于报告撰写与审阅。

---

## 1. 数据（Data）

### 1.1 Heart Disease（Cleveland）

- **来源**：UCI，`ucimlrepo.fetch_ucirepo(id=45)`  
- **任务**：二分类，标签 `y_binary = (num > 0)`（是否患病）。  
- **特征**：数值列经 **MDLP 离散化** 后与前 4 列作为 interval；分类列 one-hot，列名语义化（如 `chest_pain_4`, `thal_7`）。  
- **产物前缀**：`heart_*`（如 `heart_X_final.csv`, `heart_xgb_model.json`）。

### 1.2 Wine

- **来源**：UCI id=109，[Wine](https://archive.ics.uci.edu/dataset/109/wine)。  
- **任务**：原数据为 **3 类**；本仓库实现为 **二分类**：**是否属于 class 1**（`class == 1` 为正类）。  
  - 与论文中对该数据集的 **多类** 设定**不一致**，因此 **F1 / 规则数不宜与论文 Table 1 直接等同对比**，需在报告中单独说明。  
- **产物前缀**：`wine_*`。

### 1.3 Congressional Voting Records

- **来源**：UCI id=105，[Congressional Voting Records](https://archive.ics.uci.edu/dataset/105/congressional+voting+records)。  
- **任务**：二分类，**Republican = 1**，**Democrat = 0**。  
- **缺失**：投票列中 `?` 映射为字符串 `missing` 后参与 one-hot。  
- **产物前缀**：`voting_*`。

### 1.4 数据落盘位置

- 预处理：`data/processed/{dataset}_X_final.csv`, `{dataset}_y_binary.csv`  
- 元信息：`results/models/{dataset}_feature_meta.json`（含 `interval_features`、`target_predicate`）

---

## 2. 评估指标（Metrics）

与论文实验部分一致，对 **LIME-FOLD 学到的规则** 在测试折上计算：

| 指标 | 说明 |
|------|------|
| **Precision** | 正类（预测为 1）的精确率 |
| **Recall** | 正类召回率 |
| **Accuracy** | 准确率 |
| **F1** | 正类 F1 |
| **Avg. number of rules** | 每折 **default + abnormal** 规则数之和，再对 5 折取平均 |

**划分方式**：`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`。

**注意**：图中 **ALEPH / ALEPH+LIME / FOLD+LIME** 的数值来自**论文 Table 1 / Figure 2**，并非在本机运行 Aleph；**Your Result** 为本项目流水线输出。

---

## 3. 环境（Environment）

| 项目 | 说明 |
|------|------|
| **OS** | Windows 10（亦可在 Linux/macOS 上运行） |
| **Python** | 建议 **3.8+**（仓库在 3.8.0 下使用） |
| **虚拟环境** | 推荐项目根目录 `.venv/` |
| **依赖** | 见 `requirements.txt`（pandas, numpy, xgboost, scikit-learn, lime, matplotlib, seaborn, ucimlrepo） |

安装：

```bash
pip install -r requirements.txt
```

---

## 4. 关键实现与训练细节（Implementation & Training）

### 4.1 XGBoost（Step 1）

- 各类数据集共用 `preprocess_utils._build_xgb()` 中参数（`n_estimators=600`, `max_depth=15`, `learning_rate=0.02` 等）。  
- 在 **完整训练集** 上 `fit` 一次（与部分论文实现中“每折重训黑盒”不同，若需严格对齐可改为 CV 内重训）。

### 4.2 数值特征：MDLP 离散化

- 实现：`src/data_processing/mdlp_discretizer.py`（Fayyad & Irani 风格 MDLP）。  
- 典型参数：`min_depth=1`, `max_depth=6`, `min_samples=10`。  
- **无数值列** 的数据集（如 voting 全离散）：`preprocess_utils` 仅做 one-hot，**不进行 MDLP**。

### 4.3 LIME（Algorithm 4 前半）

- `lime.lime_tabular.LimeTabularExplainer`，`discretize_continuous=False`，特征按 categorical 处理。  
- 默认 `num_features=5`, `num_samples=5000`（可在 `run_heart_experiment.py` 用 `--num-features` 调整）。  
- 解释写入 `results/models/{dataset}_lime_explanations.pkl`。

### 4.4 Dataset Transformation + FOLD

- **E+/E−**：由 **模型预测标签** `M(r)` 构造（非仅真实标签）。  
- **BK**：由 LIME 的 `feature_weights_by_class[1]`（正类）与样本特征值生成 literal；**正/负权重均保留**（不按符号丢弃）。  
- **interval / binary**：由 `feature_meta.json` 中 `interval_features` 与列取值判定。  
- **FOLD**：`max_rule_length=5`，目标谓词见各数据集 `target_predicate`（如 `heart_disease`, `wine_target`, `voting_target`）。

### 4.5 图表

- **不生成** 论文 Figure 3 类 XGBoost 全局特征重要性图（当前默认关闭）。  
- 对比图：`results/figures/{dataset}_experiment_comparison.png`，规则数柱顶标注 **两位小数**，便于与论文 Figure 2 中小数规则数对照。

---

## 5. 与论文数值差异的常见原因

- XGBoost / LIME / FOLD 实现与原作者 Java+Prolog 栈不完全一致；  
- **Wine 二分类 vs 论文多类** 导致指标与规则数不可直接等同；  
- 5-fold 随机种子与单次划分波动；  
- MDLP 与剪枝阈值等工程参数。

更完整的讨论见项目内实验日志与 `results/logs/*.csv`。
