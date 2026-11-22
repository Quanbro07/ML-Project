"""Reproduce Table 3: representatives and reduction rate for r=1."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

SOURCE_DIR = Path(__file__).resolve().parents[2]
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from data_loader import DATASETS, load_dataset, make_preprocessor
from knn_model import KNNModel


CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

TOP_K = {
    "glass": 7,
    "iris": 3,
    "heart": 11,
    "wine": 10,
    "diabetes": 6,
    "aust": 12,
}

def evaluate_table3(dataset_name: str, n_value: int) -> dict:
    X, y = load_dataset(dataset_name)
    classifier = KNNModel(r=1, N=n_value, metric="euclidean")
    pipeline = Pipeline([
        ("preprocess", make_preprocessor(dataset_name, k=TOP_K[dataset_name])),
        ("classifier", classifier),
    ])

    result = cross_validate(
        pipeline,
        X,
        y,
        cv=CV,
        scoring="accuracy",
        return_estimator=True,
    )

    estimators = [
        estimator.named_steps["classifier"] for estimator in result["estimator"]
    ]
    return {
        "accuracy": float(np.mean(result["test_score"]) * 100),
        "avg_representatives": float(
            np.mean([estimator.n_representatives_ for estimator in estimators])
        ),
        "avg_reduction_rate": float(
            np.mean([estimator.reduction_rate_ for estimator in estimators])
        ),
    }


def build_table3() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    representatives_rows = []
    reduction_rows = []
    accuracy_rows = []

    for dataset_name in DATASETS:
        representatives_row = {"Dataset": dataset_name.capitalize()}
        reduction_row = {"Dataset": dataset_name.capitalize()}
        accuracy_row = {"Dataset": dataset_name.capitalize()}

        for n_value in range(1, 6):
            metrics = evaluate_table3(dataset_name, n_value)
            column = f"N>{n_value}"
            representatives_row[column] = metrics["avg_representatives"]
            reduction_row[column] = metrics["avg_reduction_rate"]
            accuracy_row[column] = metrics["accuracy"]

        representatives_row["kNN"] = load_dataset(dataset_name)[0].shape[0]
        representatives_rows.append(representatives_row)
        reduction_rows.append(reduction_row)
        accuracy_rows.append(accuracy_row)

    table_representatives = pd.DataFrame(representatives_rows).set_index("Dataset")
    table_reduction = pd.DataFrame(reduction_rows).set_index("Dataset")
    table_accuracy = pd.DataFrame(accuracy_rows).set_index("Dataset")

    table_representatives.loc["Average"] = table_representatives.mean(numeric_only=True)
    table_reduction.loc["Average"] = table_reduction.mean(numeric_only=True)
    table_accuracy.loc["Average"] = table_accuracy.mean(numeric_only=True)

    return table_representatives, table_reduction, table_accuracy


if __name__ == "__main__":
    print("Loading and preprocessing 6 datasets")
    representatives, reduction, accuracy = build_table3()

    print("\n=== Table 3a: Number of representatives (r=1) ===")
    print(representatives.round(2).to_string())

    print("\n=== Table 3b: Average reduction rate (%) (r=1) ===")
    print(reduction.round(2).to_string())

    print("\n=== Evaluation accuracy (%) (r=1) ===")
    print(accuracy.round(2).to_string())
