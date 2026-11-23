"""Reproduce Table 3 with PyTorch MLP Latent Feature Extractor & Visualization."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

SOURCE_DIR = Path(__file__).resolve().parents[2]
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from data_loader import DATASETS, load_dataset, make_preprocessor
from knn_model import KNNModel


CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

TOP_K = {
    "glass": "all",
    "iris": 3,
    "heart": 11,
    "wine": 10,
    "diabetes": 6,
    "aust": 12,
}


class _MLPNet(nn.Module):
    """Mạng MLP đơn giản để học biểu diễn ẩn (latent space)."""

    def __init__(self, in_features: int, latent_dim: int, num_classes: int):
        super().__init__()
        hidden_dim = max(latent_dim * 2, 16)
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        logits = self.head(latent)
        return logits, latent


class TorchLatentExtractor(BaseEstimator, TransformerMixin):
    """Transformer chuẩn sklearn: train MLP trên tập train và xuất vector latent."""

    def __init__(
        self,
        latent_dim: int = 8,
        epochs: int = 80,
        lr: float = 1e-2,
        weight_decay: float = 1e-3,
        seed: int = 42,
    ):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        in_features = X.shape[1]
        num_classes = int(np.max(y)) + 1

        self.model_ = _MLPNet(in_features, self.latent_dim, num_classes)
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        self.model_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits, _ = self.model_(X_tensor)
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()

        self.model_.eval()
        with torch.no_grad():
            _, train_latent = self.model_(X_tensor)
        self.scaler_ = MinMaxScaler()
        self.scaler_.fit(train_latent.numpy())
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            _, latent = self.model_(X_tensor)
        return self.scaler_.transform(latent.numpy())


def evaluate_table3(
    dataset_name: str,
    n_value: int,
    use_latent: bool = True,
    epochs: int = 80,
) -> dict:
    X, y = load_dataset(dataset_name)
    classifier = KNNModel(r=1, N=n_value, metric="euclidean")

    steps = [("preprocess", make_preprocessor(dataset_name, k=TOP_K[dataset_name]))]
    if use_latent:
        steps.append(
            ("latent_features", TorchLatentExtractor(latent_dim=8, epochs=epochs, seed=42))
        )
    steps.append(("classifier", classifier))

    pipeline = Pipeline(steps)

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


def build_table3(
    use_latent: bool = True, epochs: int = 80
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    representatives_rows = []
    reduction_rows = []
    accuracy_rows = []

    for dataset_name in DATASETS:
        representatives_row = {"Dataset": dataset_name.capitalize()}
        reduction_row = {"Dataset": dataset_name.capitalize()}
        accuracy_row = {"Dataset": dataset_name.capitalize()}

        for n_value in range(1, 6):
            metrics = evaluate_table3(
                dataset_name, n_value, use_latent=use_latent, epochs=epochs
            )
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


def plot_table3_results(
    acc_latent: pd.DataFrame,
    red_latent: pd.DataFrame,
    acc_euclidean: pd.DataFrame | None = None,
    red_euclidean: pd.DataFrame | None = None,
    sample_dataset: str = "aust",
    epochs: int = 80,
) -> None:
    """
    Vẽ 3 biểu đồ tổng hợp:
    1. So sánh đường cong Accuracy (Euclidean gốc vs. MLP Latent Space) qua N>1..N>5.
    2. Tỷ lệ nén trung bình (%) qua N>1..N>5.
    3. Trực quan hóa 2D (PCA) phân bố cụm trên tập mẫu (Euclidean vs. Latent Space).
    """
    cols = ["N>1", "N>2", "N>3", "N>4", "N>5"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # --- Subplot 1: Average Accuracy Curve ---
    ax1 = axes[0]
    ax1.plot(
        cols,
        acc_latent.loc["Average", cols],
        marker="o",
        linewidth=2.5,
        color="#2563eb",
        label=f"MLP Latent Space ({epochs} epochs)",
    )
    if acc_euclidean is not None:
        ax1.plot(
            cols,
            acc_euclidean.loc["Average", cols],
            marker="s",
            linestyle="--",
            linewidth=2.2,
            color="#dc2626",
            label="Raw Euclidean Space",
        )
    ax1.set_title("Average Accuracy vs. Pruning Threshold", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Pruning Threshold (N > i)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # --- Subplot 2: Reduction Rate ---
    ax2 = axes[1]
    ax2.plot(
        cols,
        red_latent.loc["Average", cols],
        marker="^",
        linewidth=2.5,
        color="#059669",
        label="MLP Latent Space",
    )
    if red_euclidean is not None:
        ax2.plot(
            cols,
            red_euclidean.loc["Average", cols],
            marker="d",
            linestyle="--",
            linewidth=2.2,
            color="#d97706",
            label="Raw Euclidean Space",
        )
    ax2.set_title("Average Reduction Rate (%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Pruning Threshold (N > i)")
    ax2.set_ylabel("Reduction Rate (%)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    # --- Subplot 3: 2D PCA Projection of Latent Space on Sample Dataset ---
    ax3 = axes[2]
    X_raw, y_raw = load_dataset(sample_dataset)
    prep = make_preprocessor(sample_dataset, k=TOP_K[sample_dataset])
    X_prep = prep.fit_transform(X_raw, y_raw)

    extractor = TorchLatentExtractor(latent_dim=8, epochs=epochs, seed=42)
    X_lat = extractor.fit_transform(X_prep, y_raw)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_lat)

    scatter = ax3.scatter(
        X_2d[:, 0], X_2d[:, 1], c=y_raw, cmap="coolwarm", alpha=0.8, edgecolors="k", s=38
    )
    ax3.set_title(
        f"2D PCA of MLP Latent Space ({sample_dataset.capitalize()})",
        fontsize=12,
        fontweight="bold",
    )
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.grid(True, linestyle=":", alpha=0.6)
    fig.colorbar(scatter, ax=ax3, label="Class Label")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Evaluating baseline Raw Euclidean Space...")
    rep_euc, red_euc, acc_euc = build_table3(use_latent=False)

    print("Evaluating MLP Latent Space (epochs=80)...")
    representatives, reduction, accuracy = build_table3(use_latent=True, epochs=80)

    try:
        from IPython.display import display
    except ImportError:
        display = print

    print("\n=== Table 3a: Number of representatives (r=1) ===")
    display(representatives.round(2))

    print("\n=== Table 3b: Average reduction rate (%) (r=1) ===")
    display(reduction.round(2))

    print("\n=== Evaluation accuracy (%) (r=1) ===")
    display(accuracy.round(2))

    plot_table3_results(
        acc_latent=accuracy,
        red_latent=reduction,
        acc_euclidean=acc_euc,
        red_euclidean=red_euc,
        sample_dataset="aust",
        epochs=80,
    )