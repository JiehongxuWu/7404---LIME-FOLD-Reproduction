import numpy as np
from ucimlrepo import fetch_ucirepo

from .preprocess_utils import preprocess_and_train


def main():
    wine = fetch_ucirepo(id=109)
    X = wine.data.features
    y = wine.data.targets

    # convert 3-class target to binary to fit current LIME-FOLD pipeline
    y_col = y.columns[0]
    y_binary = (y[y_col].astype(int) == 1).astype(np.int64)

    numeric_cols = X.columns.tolist()
    categorical_cols = []

    preprocess_and_train(
        dataset_name="wine",
        X=X,
        y_binary=y_binary,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        numeric_rename={c: c.lower() for c in numeric_cols},
    )


if __name__ == "__main__":
    main()

