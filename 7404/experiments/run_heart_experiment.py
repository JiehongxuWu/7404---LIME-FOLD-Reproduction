#!/usr/bin/env python
"""
Heart Disease 数据集完整复现脚本
按照论文流程逐步执行：
Step1: 预处理 + 训练 XGBoost（已在 heart_preprocessing.py 中完成并落盘）
Step2: 基于 Step1 产物生成 LIME explanations
Step3: 运行 LIME-FOLD 实验
Step4: 保存结果
Step5: 画图与论文对比
"""

import pickle
from pathlib import Path

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is on sys.path so `import src...` works when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explanation.lime_explanations import (
    DEFAULT_EXPLANATIONS_PATH,
    generate_lime_explanations,
    load_step1_outputs,
)
from src.models.lime_fold_main import run_lime_fold_experiment


RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = RESULTS_DIR / "figures"
for d in [RESULTS_DIR, MODELS_DIR, LOGS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def main():
    # Step 1: 加载预处理后的数据与模型
    print("Step 1: 加载预处理后的数据与 XGBoost 模型...")
    X_final, y_binary, model, feature_names = load_step1_outputs()

    # Step 2: 生成 / 加载 LIME 解释
    print("\nStep 2: 生成 LIME 解释...")
    explanations_path = DEFAULT_EXPLANATIONS_PATH
    if not explanations_path.exists():
        explanations = generate_lime_explanations(
            X_final, y_binary, model, feature_names, output_file=explanations_path
        )
    else:
        print(f"LIME 解释已存在，直接从 {explanations_path} 加载...")
        with explanations_path.open("rb") as f:
            explanations = pickle.load(f)

    # Step 3: 运行 LIME-FOLD 实验
    print("\nStep 3: 运行 LIME-FOLD 实验...")
    fold_model, results = run_lime_fold_experiment(
        X_final,
        y_binary,
        explanations,
        use_cross_validation=True,
        n_splits=5,
        random_state=42,
    )

    # Step 4: 保存结果
    print("\nStep 4: 保存实验结果...")
    results_df = pd.DataFrame([results])
    results_csv = LOGS_DIR / "lime_fold_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"实验结果已保存到 {results_csv}")

    # Step 5: 可视化对比
    print("\nStep 5: 生成结果对比图...")

    paper_results = {
        "ALEPH": [0.76, 0.75, 0.78, 0.75, 18],
        "ALEPH+LIME": [0.79, 0.70, 0.79, 0.74, 12],
        # Table 1 (heart): Prec 0.82, Recall 0.74, Acc 0.82, F1 0.78
        "FOLD+LIME": [0.82, 0.74, 0.82, 0.78, 6],
    }

    your_results = [
        results["precision"],
        results["recall"],
        results["accuracy"],
        results["f1"],
        results.get("avg_num_rules", np.nan),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = ["ALEPH", "ALEPH+LIME", "FOLD+LIME", "Your Result"]
    f1_scores = [
        paper_results["ALEPH"][3],
        paper_results["ALEPH+LIME"][3],
        paper_results["FOLD+LIME"][3],
        results["f1"],
    ]
    colors = ["skyblue", "lightgreen", "lightcoral", "gold"]
    axes[0].bar(methods, f1_scores, color=colors)
    axes[0].set_ylabel("F1 Score")
    axes[0].set_title("F1 Score Comparison")
    # Ensure "Your Result" bar is visible even if it's below 0.6
    ymin = min(f1_scores + [0.0])
    axes[0].set_ylim([max(0.0, ymin - 0.05), 1.0])
    for i, v in enumerate(f1_scores):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center")

    rules_count = [
        paper_results["ALEPH"][4],
        paper_results["ALEPH+LIME"][4],
        paper_results["FOLD+LIME"][4],
        results.get("avg_num_rules", np.nan),
    ]
    axes[1].bar(methods, rules_count, color=colors)
    axes[1].set_ylabel("Number of Rules")
    axes[1].set_title("Number of Rules Comparison")
    for i, v in enumerate(rules_count):
        axes[1].text(i, v + 0.5, str(int(v)), ha="center")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "experiment_comparison.png"
    plt.savefig(fig_path, dpi=150)
    plt.show()

    print("\n" + "=" * 60)
    print("复现完成！")
    print("=" * 60)
    print(f"您的 F1 分数: {results['f1']:.4f} (论文 FOLD+LIME: 0.78)")
    if "avg_num_rules" in results:
        print(f"您的平均规则数量: {results['avg_num_rules']:.2f} (论文: ~6-8)")
    elif "num_rules" in results:
        print(f"您的规则数量: {results['num_rules']} (论文: ~6-8)")
    print(f"结果保存在: {fig_path}")


if __name__ == "__main__":
    main()