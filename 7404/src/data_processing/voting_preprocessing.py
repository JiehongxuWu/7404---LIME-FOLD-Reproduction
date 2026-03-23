"""
Congressional Voting Records (UCI id=105)
https://archive.ics.uci.edu/dataset/105/congressional+voting+records
二分类：Republican=1，Democrat=0；特征为 y/n/?，? 视为缺失后 one-hot。
"""
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from .preprocess_utils import preprocess_and_train


def main():
    voting = fetch_ucirepo(id=105)
    X = voting.data.features
    y = voting.data.targets

    y_col = y.columns[0]
    labels = y[y_col].astype(str).str.strip().str.lower()
    y_binary = (labels == "republican").astype(np.int64)

    X_clean = X.copy()
    for c in X_clean.columns:
        X_clean[c] = (
            X_clean[c]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"?": "missing", "nan": "missing"})
        )

    categorical_cols = X_clean.columns.tolist()
    preprocess_and_train(
        dataset_name="voting",
        X=X_clean,
        y_binary=y_binary,
        numeric_cols=[],
        categorical_cols=categorical_cols,
    )


if __name__ == "__main__":
    main()
