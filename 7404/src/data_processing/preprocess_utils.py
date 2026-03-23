import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer

from .mdlp_discretizer import MDLPDiscretizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "results" / "models"


def _build_xgb():
    return xgb.XGBClassifier(
        objective="binary:logistic",
        importance_type="weight",
        random_state=42,
        n_estimators=600,
        max_depth=15,
        learning_rate=0.02,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_lambda=3,
        reg_alpha=0.2,
        min_child_weight=2,
        eval_metric="logloss",
        use_label_encoder=False,
    )


def preprocess_and_train(
    dataset_name: str,
    X: pd.DataFrame,
    y_binary: pd.Series,
    numeric_cols,
    categorical_cols,
    numeric_rename=None,
    categorical_rename=None,
):
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    numeric_cols = [c for c in numeric_cols if c in X.columns]
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    numeric_rename = numeric_rename or {}
    categorical_rename = categorical_rename or {}

    X_work = X.copy()

    if numeric_cols:
        # numeric coercion first, then impute
        for c in numeric_cols:
            X_work[c] = pd.to_numeric(X_work[c], errors="coerce")

        imputer = SimpleImputer(strategy="median")
        X_num = pd.DataFrame(imputer.fit_transform(X_work[numeric_cols]), columns=numeric_cols, index=X_work.index)

        mdlp = MDLPDiscretizer(min_depth=1, max_depth=6, min_samples=10)
        X_num_arr = X_num.to_numpy(dtype=np.float64)
        X_num_discrete_arr = mdlp.fit_transform(X_num_arr, y_binary.to_numpy())
        X_num_discrete = pd.DataFrame(X_num_discrete_arr.astype(np.int64), columns=numeric_cols, index=X_num.index)
        X_num_discrete.rename(columns={k: v for k, v in numeric_rename.items() if k in X_num_discrete.columns}, inplace=True)
    else:
        X_num_discrete = pd.DataFrame(index=X_work.index)

    if categorical_cols:
        X_cat = X_work[categorical_cols].copy().fillna("missing").astype(str)
        X_onehot = pd.get_dummies(X_cat, columns=categorical_cols, prefix=categorical_cols, drop_first=False)
        if categorical_rename:
            X_onehot.rename(columns=categorical_rename, inplace=True)
    else:
        X_onehot = pd.DataFrame(index=X_work.index)

    X_final = pd.concat([X_num_discrete, X_onehot], axis=1)

    model = _build_xgb()
    model.fit(X_final, y_binary)

    (DATA_PROCESSED_DIR / f"{dataset_name}_X_final.csv").write_text("", encoding="utf-8")
    X_final.to_csv(DATA_PROCESSED_DIR / f"{dataset_name}_X_final.csv", index=False)
    pd.Series(y_binary, name="y_binary").to_csv(DATA_PROCESSED_DIR / f"{dataset_name}_y_binary.csv", index=False)
    model.save_model(str(MODELS_DIR / f"{dataset_name}_xgb_model.json"))

    feature_meta = {
        "dataset_name": dataset_name,
        "interval_features": X_num_discrete.columns.tolist(),
        "target_predicate": f"{dataset_name}_target",
    }
    with (MODELS_DIR / f"{dataset_name}_feature_meta.json").open("w", encoding="utf-8") as f:
        json.dump(feature_meta, f, ensure_ascii=False, indent=2)

    print(f"[{dataset_name}] saved X_final/model/meta.")

