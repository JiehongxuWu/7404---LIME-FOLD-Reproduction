import pickle
from pathlib import Path

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
DEFAULT_EXPLANATIONS_PATH = MODELS_DIR / "lime_explanations.pkl"


def load_step1_outputs():
    """从 CSV / JSON 加载 Step1 产物：X_final, y_binary, model, feature_names"""
    X_final = pd.read_csv(DATA_PROCESSED_DIR / "X_final.csv")
    y_binary = pd.read_csv(DATA_PROCESSED_DIR / "y_binary.csv")["y_binary"]

    model = xgb.XGBClassifier()
    model.load_model(str(MODELS_DIR / "xgb_model.json"))

    feature_names = X_final.columns.tolist()
    return X_final, y_binary, model, feature_names


def generate_lime_explanations(X_final, y_binary, model, feature_names, output_file: Path = DEFAULT_EXPLANATIONS_PATH):
    """
    为每个样本生成LIME解释（对应论文 Algorithm 4 前半部分：LIME 生成 transformed dataset）
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Algorithm 4 expects explanation elements as either:
    # - an interval feature (discretized numeric) or
    # - an equality expression f_v = 0/1 for categorical features
    # We treat ALL features as categorical so LIME outputs equality-like conditions consistently.
    X_np = X_final.values.astype(float)
    categorical_features = list(range(X_np.shape[1]))
    categorical_names = {}
    for j, name in enumerate(feature_names):
        uniq = sorted(pd.unique(X_final.iloc[:, j]).tolist())
        categorical_names[j] = [str(u) for u in uniq]

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_np,
        feature_names=feature_names,
        class_names=["No Disease", "Heart Disease"],
        discretize_continuous=False,
        categorical_features=categorical_features,
        categorical_names=categorical_names,
        random_state=42,
    )

    explanations = {}

    for i in range(len(X_final)):
        x_row = X_final.iloc[i].values.astype(float)
        # Algorithm 4 uses M(r) (model's label) for E+/E- construction
        model_pred_label = int(model.predict(x_row.reshape(1, -1))[0])

        exp = explainer.explain_instance(
            x_row,
            model.predict_proba,
            num_features=5,  # 选择 top-5 特征
            num_samples=5000,
        )

        weights_by_class = exp.as_map()
        # Prefer to use the class that the model predicted (Algorithm 4: M(r)).
        # Some LIME configurations may not include both class keys; fallback safely.
        if model_pred_label in weights_by_class:
            chosen_weights = weights_by_class[model_pred_label]
        elif 1 in weights_by_class:
            chosen_weights = weights_by_class[1]
        else:
            # last resort: pick any available label
            chosen_weights = next(iter(weights_by_class.values()))

        explanations[f"inst_{i}"] = {
            # Stable mapping for Algorithm 4
            "feature_weights": chosen_weights,
            "feature_weights_by_class": weights_by_class,
            "features": exp.as_list(),
            "label": int(y_binary.iloc[i] if hasattr(y_binary, "iloc") else y_binary[i]),
            "model_label": model_pred_label,
        }

        if i % 50 == 0:
            print(f"Processed {i}/{len(X_final)} samples")

    with output_file.open("wb") as f:
        pickle.dump(explanations, f)

    return explanations


if __name__ == "__main__":
    X_final, y_binary, model, feature_names = load_step1_outputs()
    explanations = generate_lime_explanations(X_final, y_binary, model, feature_names)
    print(f"生成了 {len(explanations)} 个 LIME 解释，已保存到 {DEFAULT_EXPLANATIONS_PATH}")