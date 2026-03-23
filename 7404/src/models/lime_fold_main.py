import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score
from .fold_algorithm import FOLD

def _literal_from_feature_value(feature_name: str, value, is_binary: bool, is_interval_feature: bool):
    """
    Algorithm 4 对应的“解释语言”生成：
    - 离散数值特征：feature_name_bin{n}
    - 0/1 特征：value==1 -> feature_name, value==0 -> -feature_name（classical negation）
    """
    # 处理 NaN（理论上 Step1 已填补，不应出现）
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    # interval feature (Algorithm 4 line 11-13): f(r.id, n)
    # IMPORTANT: even if an interval feature has only 2 bins {0,1}, it is still an interval feature,
    # and should be represented as f_bin{n}, NOT as classical negation.
    if is_interval_feature:
        try:
            return f"{feature_name}_bin{int(value)}"
        except Exception:
            return None

    # one-hot / binary (Algorithm 4 line 14-20): f_v=0/1 with classical negation
    if is_binary and (value in (0, 1) or (isinstance(value, (np.integer, int)) and int(value) in (0, 1))):
        return feature_name if int(value) == 1 else f"-{feature_name}"

    # fallback: stringify
    return f"{feature_name}={value}"


def transform_dataset_with_lime(explanations, X_final, interval_feature_names=None):
    """
    用LIME解释转换数据集（论文Algorithm 4）
    """
    bk = {}  # 背景知识
    E_plus = []  # 正例
    E_minus = []  # 负例

    feature_names = X_final.columns.tolist()
    if interval_feature_names:
        interval_features = set(interval_feature_names)
    else:
        # backward-compatible fallback for heart pipeline
        interval_features = set(feature_names[:4])
    # 判断哪些列是真正二值（one-hot）特征：unique ⊆ {0,1}
    binary_cols = set()
    for col in feature_names:
        try:
            uniq = set(pd.unique(X_final[col]))
        except Exception:
            continue
        if uniq.issubset({0, 1, 0.0, 1.0, np.int64(0), np.int64(1)}):
            binary_cols.add(col)

    for inst_id, exp_data in explanations.items():
        # Algorithm 4: use M(r) (model label), not ground-truth label, to construct E+/E-
        m_label = exp_data.get("model_label", exp_data.get("label", 0))
        if m_label == 1:
            E_plus.append(inst_id)
        else:
            E_minus.append(inst_id)

        # 构建该样本的背景知识
        bk[inst_id] = {}

        # Algorithm 4: explanation = LIME(M, r)
        # 对每个 (e, w) ∈ explanation：
        #   - 如果 e 是 interval feature：BK += f(r.id, n)
        #   - 如果 e 是 equality expr f_v = 0：BK += -f(r.id, v)  (classical negation)
        #   - 如果 e 是 equality expr f_v = 1：BK += f(r.id, v)
        #
        # 这里不解析字符串表达式，而是使用保存的 (feature_index, weight)，
        # 再用该样本的 X_final 取值决定 literal 形式。
        # Align with Algorithm 4: explanation pairs include both positive and negative weights
        # for the *target concept* we are learning (here: Heart Disease / positive class).
        # Therefore, always prefer the weights of the positive class (class id = 1).
        fw = None
        weights_by_class = exp_data.get("feature_weights_by_class")
        if isinstance(weights_by_class, dict) and 1 in weights_by_class:
            fw = weights_by_class[1]
        else:
            # fallback to whatever was stored historically
            fw = exp_data.get("feature_weights")
        if fw is None:
            # fallback compatibility
            for feature_str, weight in exp_data.get("features", []):
                lit = feature_str if weight > 0 else f"-{feature_str}"
                bk[inst_id][lit] = 1
            continue

        inst_i = int(inst_id.split("_", 1)[1])
        row = X_final.iloc[inst_i]

        for feat_idx, weight in fw:
            # Algorithm 4 (paper lines 492-503): retrieve negative-weight features too.
            # In the transformed dataset, we store *all* top-K features returned by LIME,
            # and use the sample's actual 0/1 (or interval-bin) value to decide whether
            # the literal is f(...) or -f(...) (classical negation).
            feat_idx = int(feat_idx)
            feat_name = feature_names[feat_idx]
            val = row.iloc[feat_idx]

            lit = _literal_from_feature_value(
                feat_name,
                val,
                is_binary=(feat_name in binary_cols),
                is_interval_feature=(feat_name in interval_features),
            )
            if lit is None:
                continue

            bk[inst_id][lit] = 1

    return bk, E_plus, E_minus

def run_lime_fold_experiment(
    X_final,
    y_binary,
    explanations,
    interval_feature_names=None,
    target_predicate: str = "target",
    dataset_name: str = "dataset",
    test_size=0.3,
    use_cross_validation: bool = True,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    运行LIME-FOLD实验，与论文表1对比
    """
    # Algorithm 4: 先对全体样本做 transformed dataset
    bk_all, E_plus_all, E_minus_all = transform_dataset_with_lime(
        explanations,
        X_final,
        interval_feature_names=interval_feature_names,
    )

    def _eval_on_ids(fold_model: FOLD, ids):
        y_pred_local = []
        for inst_id in sorted(ids, key=lambda s: int(s.split("_", 1)[1])):
            pred = fold_model.predict(inst_id, bk_all)
            y_pred_local.append(1 if pred else 0)

        idxs = [int(s.split("_", 1)[1]) for s in sorted(ids, key=lambda s: int(s.split("_", 1)[1]))]
        y_true_local = y_binary.iloc[idxs] if hasattr(y_binary, "iloc") else np.array([y_binary[i] for i in idxs])

        y_pred_arr = np.array(y_pred_local)
        y_true_arr = np.array(y_true_local)
        tp = int(np.sum((y_pred_arr == 1) & (y_true_arr == 1)))
        pred_pos = int(np.sum(y_pred_arr == 1))
        actual_pos = int(np.sum(y_true_arr == 1))

        precision_local = tp / (pred_pos + 1e-10)
        recall_local = tp / (actual_pos + 1e-10)
        accuracy_local = accuracy_score(y_true_arr, y_pred_arr)
        f1_local = f1_score(y_true_arr, y_pred_arr)
        return precision_local, recall_local, accuracy_local, f1_local

    fold_metrics = []
    fold_models = []

    if use_cross_validation:
        # Paper protocol: 5-fold cross validation (Experiments section).
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        X_idx = np.arange(len(X_final))
        y_arr = y_binary.to_numpy() if hasattr(y_binary, "to_numpy") else np.array(y_binary)

        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_idx, y_arr), start=1):
            train_ids = {f"inst_{i}" for i in train_idx}
            test_ids = {f"inst_{i}" for i in test_idx}

            E_plus_train = [i for i in E_plus_all if i in train_ids]
            E_minus_train = [i for i in E_minus_all if i in train_ids]
            bk_train = {i: bk_all[i] for i in train_ids}

            print(f"\n训练FOLD模型... (fold {fold_i}/{n_splits})")
            fold_model = FOLD(max_rule_length=5)
            fold_model.fit(target_predicate, E_plus_train, E_minus_train, bk_train)
            fold_models.append(fold_model)

            precision, recall, accuracy, f1 = _eval_on_ids(fold_model, test_ids)
            num_rules = len(getattr(fold_model, "default_rules", [])) + len(getattr(fold_model, "ab_rules", []))
            fold_metrics.append(
                {
                    "precision": precision,
                    "recall": recall,
                    "accuracy": accuracy,
                    "f1": f1,
                    "num_rules": num_rules,
                }
            )
            print(f"fold {fold_i}: Precision={precision:.4f} Recall={recall:.4f} Accuracy={accuracy:.4f} F1={f1:.4f} Rules={num_rules}")

        precision = float(np.mean([m["precision"] for m in fold_metrics]))
        recall = float(np.mean([m["recall"] for m in fold_metrics]))
        accuracy = float(np.mean([m["accuracy"] for m in fold_metrics]))
        f1 = float(np.mean([m["f1"] for m in fold_metrics]))
        avg_rules = float(np.mean([m["num_rules"] for m in fold_metrics]))

        print("\n" + "=" * 50)
        print(f"{n_splits}-fold 交叉验证平均结果：")
        print("=" * 50)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"平均规则数量: {avg_rules:.2f}")

        # 打印最后一折学习到的规则（用于快速对照论文示例规则）
        fold_model = fold_models[-1]
        print("\n" + "=" * 50)
        print("最后一折（last fold）学习到的规则：")
        print("=" * 50)
        fold_model.print_rules()

    else:
        # Backward-compatible single split (not the paper protocol).
        indices = np.arange(len(X_final))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=y_binary
        )
        train_ids = {f"inst_{i}" for i in train_idx}
        test_ids = {f"inst_{i}" for i in test_idx}

        E_plus_train = [i for i in E_plus_all if i in train_ids]
        E_minus_train = [i for i in E_minus_all if i in train_ids]
        bk_train = {i: bk_all[i] for i in train_ids}

        print("训练FOLD模型...")
        fold_model = FOLD(max_rule_length=5)
        fold_model.fit(target_predicate, E_plus_train, E_minus_train, bk_train)

        print("\n" + "=" * 50)
        print("LIME-FOLD学习到的规则：")
        print("=" * 50)
        fold_model.print_rules()

        precision, recall, accuracy, f1 = _eval_on_ids(fold_model, test_ids)
        num_rules = len(getattr(fold_model, "default_rules", [])) + len(getattr(fold_model, "ab_rules", []))
        avg_rules = float(num_rules)

        print("\n" + "=" * 50)
        print("测试集结果：")
        print("=" * 50)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"规则数量: {num_rules}")

    if str(dataset_name).lower() == "heart":
        print("\n" + "=" * 50)
        print("论文表1结果 (Heart数据集)：")
        print("=" * 50)
        paper_results = {
            "ALEPH": {"Precision": 0.76, "Recall": 0.75, "Accuracy": 0.78, "F1": 0.75},
            "ALEPH+LIME": {"Precision": 0.79, "Recall": 0.70, "Accuracy": 0.79, "F1": 0.74},
            "FOLD+LIME": {"Precision": 0.82, "Recall": 0.74, "Accuracy": 0.82, "F1": 0.78},
        }
        for method, metrics in paper_results.items():
            print(
                f"{method:12s}: Precision={metrics['Precision']:.2f}, "
                f"Recall={metrics['Recall']:.2f}, "
                f"Accuracy={metrics['Accuracy']:.2f}, "
                f"F1={metrics['F1']:.2f}"
            )

    out = {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "avg_num_rules": avg_rules,
    }
    if fold_metrics:
        out["fold_metrics"] = fold_metrics

    return fold_model, out