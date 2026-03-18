## Reproduction of the LIME-FOLD Algorithm (Heart Disease Dataset)

This project reproduce the AAAI-19 paper. 
**“Induction of Non-Monotonic Logic Programs to Explain Boosted Tree Models Using LIME”**  
The LIME-FOLD experimental procedure and results on the UCI Heart Disease (Cleveland) dataset (heart row in Table 1).
---

## program structure

- **`src/`**：源码
  - **`data_processing/`**
    - `heart_preprocessing.py`：Step 1，下载 UCI heart 数据、预处理、MDLP 离散化、训练 XGBoost，并落盘：
      - `data/processed/X_final.csv`
      - `data/processed/y_binary.csv`
      - `results/models/xgb_model.json`
      - `results/figures/xgb_feature_importance.png`（Step1 特征重要性图）
    - `mdlp_discretizer.py`：Fayyad & Irani (1993) MDLP 离散化实现
  - **`explanation/`**
    - `lime_explanations.py`：基于 Step1 的 XGBoost 模型，对每个样本调用 LIME，生成解释并保存：
      - `results/models/lime_explanations.pkl`
  - **`models/`**
    - `fold_algorithm.py`：FOLD 算法实现（对应论文 Algorithm 3 + FOIL IG 公式）
    - `lime_fold_main.py`：LIME‑FOLD 主流程：
      - 根据论文 Algorithm 4，将 LIME 解释 + 预处理数据转成 ILP 背景知识 `BK`、正负例 `E+ / E-`
      - 在该 transformed dataset 上运行 FOLD，并使用 **5‑fold 交叉验证**评估 Precision / Recall / Accuracy / F1
- **`experiments/`**
  - `run_heart_experiment.py`：完整复现实验入口脚本（Step1–Step5，一键运行）
- **`data/`**
  - `raw/`：原始数据（未纳入版本控制）
  - `processed/`：预处理结果（由 Step1 自动生成）
- **`results/`**
  - `models/`：XGBoost 模型、LIME 解释等中间产物
  - `figures/`：实验图（Step1 特征重要性；表 1 对比图）
  - `logs/`：实验数值结果（CSV）

---

## 环境与依赖

推荐使用虚拟环境（项目根目录已经假定 `.venv/`）：

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

`requirements.txt` 中包含：

- XGBoost、scikit‑learn、pandas、numpy
- LIME（解释器）
- `ucimlrepo`（自动下载 UCI heart 数据）
- MDLP 离散化所需依赖

---

## 运行步骤（完整复现实验）

所有命令默认在项目根目录执行，已激活虚拟环境。

1. **Step 1：预处理 + 训练 XGBoost + 保存模型**

   ```bash
   python -m src.data_processing.heart_preprocessing
   ```

   该脚本会：

   - 从 UCI 仓库下载 Heart Disease (Cleveland) 数据集；
   - 使用中位数填补缺失值；
   - 对数值特征（`blood_pressure`, `serum_cholestoral`, `oldpeak`, `major_vessels`）进行 MDLP 离散化；
   - 对分类特征做 one‑hot 并重命名为论文中使用的语义名（如 `chest_pain_4`, `thal_7` 等）；
   - 在完整数据上训练一个 XGBoost 二分类模型；
   - 保存预处理结果与模型；绘制并保存特征重要性图：
     - `results/figures/xgb_feature_importance.png`。

2. **Step 2–5：生成 LIME 解释 + LIME‑FOLD + 与论文对比**

   ```bash
   python experiments/run_heart_experiment.py
   ```

   该脚本会按论文流程自动完成：

   - **Step 2**：如果不存在 `results/models/lime_explanations.pkl`，则调用
     `generate_lime_explanations` 基于 Step1 的 XGBoost 模型为每个样本生成 LIME 解释；
   - **Step 3**：调用 `run_lime_fold_experiment`：
     - 按论文 Algorithm 4 将 LIME 解释转换为 ILP 背景知识 `BK` 与正/负例 `E+, E-`；
     - 在 transformed dataset 上运行 FOLD（Algorithm 3），学习默认规则与异常规则；
     - 使用 **5‑fold Stratified Cross‑Validation** 计算平均 Precision / Recall / Accuracy / F1；
   - **Step 4**：将本次实验的数值结果保存为
     `results/logs/lime_fold_results.csv`；
   - **Step 5**：生成与论文表 1 对比的柱状图：
     - `results/figures/experiment_comparison.png`  
       （左图为 F1 对比，右图为规则数量对比）。

脚本执行结束后，终端会输出：

- 本次 5‑fold 平均的 Precision / Recall / Accuracy / F1；
- 平均规则数量（与论文中 heart 数据集约 6 条规则对比）；
- 保存的图像与日志文件路径。

---

## 当前复现程度说明

- 实现上对齐了论文中的关键伪代码：
  - FOIL 信息增益与 FOLD 算法（Algorithm 3）；
  - 使用 LIME 生成局部解释并按 Algorithm 4 进行数据集转换（包含正/负权重特征，数值特征采用 MDLP 离散化）。
- 评估方式采用与论文一致的 **5‑fold cross‑validation**，并输出平均指标。
- 在 Heart Disease 数据集上的实验结果（F1 与规则数量）已经与论文表 1 中的 FOLD+LIME 行较为接近，但未必完全数值一致（可能受到：
  - XGBoost 具体超参数、
  - MDLP 实现与参数、
  - 随机性与折分方式  
等因素影响）。

本仓库主要侧重于：**复现算法流程与可解释规则形态**，并尽量接近论文报告的指标，方便教学与进一步实验。
