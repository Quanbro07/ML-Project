"""Load the six classification datasets in Guo et al. (2003), Table 1."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.metrics import mutual_info_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder


DATASETS = ("glass", "iris", "heart", "wine", "diabetes", "aust")
_DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
_UCI_DATASETS = {
    "glass": (42, "Glass.csv"),
    "iris": (53, "Iris.csv"),
    "heart": (145, "Heart.csv"),
    "wine": (109, "Wine.csv"),
    "aust": (143, "Aust.csv"),
}
_CATEGORICAL = {
    "heart": (2, 6, 12),
    "aust": (3, 4, 5, 11),
}
_SHAPES = {
    "glass": (214, 9),
    "iris": (150, 4),
    "heart": (270, 13),
    "wine": (178, 13),
    "diabetes": (768, 8),
    "aust": (690, 14),
}


def _dataset_name(name: str) -> str:
    name = name.strip().lower()
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {', '.join(DATASETS)}.")
    return name


def load_dataset(
    name: str, *, data_home: str | Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return float features and integer class labels as NumPy arrays.

    Read CSVs from the project's Data directory unless data_home is supplied.
    Missing UCI CSVs are downloaded and saved on first use. Diabetes.csv
    must exist locally. Fit make_preprocessor(name) on the training data.
    """
    name = _dataset_name(name)
    directory = _DATA_DIR if data_home is None else Path(data_home)
    if name == "diabetes":
        frame = pd.read_csv(directory / "Diabetes.csv", comment="#", header=None)
    else:
        uci_id, filename = _UCI_DATASETS[name]
        path = directory / filename
        frame = pd.read_csv(path) if path.is_file() else None
        if frame is None:
            from ucimlrepo import fetch_ucirepo

            data = fetch_ucirepo(id=uci_id).data
            frame = pd.concat([data.features, data.targets], axis=1)
            if frame.shape != (_SHAPES[name][0], _SHAPES[name][1] + 1):
                raise ValueError(f"{name}: unexpected UCI dataset shape {frame.shape}.")
            directory.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                dir=directory, mode="w", encoding="utf-8", delete_on_close=False
            ) as temporary:
                frame.to_csv(temporary, index=False)
                temporary.close()
                Path(temporary.name).replace(path)

    features, target = frame.iloc[:, :-1], frame.iloc[:, -1]
    X = features.to_numpy(dtype=np.float64)
    y = LabelEncoder().fit_transform(target)
    if X.shape != _SHAPES[name]:
        raise ValueError(f"{name}: expected features of shape {_SHAPES[name]}, got {X.shape}.")
    return X, y


def _information_gain(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Estimate information gain using up to ten quantile bins per feature."""
    scores = []
    for column in X.T:
        values, codes = np.unique(column, return_inverse=True)
        if values.size > 10:
            edges = np.unique(np.quantile(column, np.arange(1, 10) / 10))
            codes = np.searchsorted(edges, column)
        scores.append(mutual_info_score(y, codes))
    return np.asarray(scores)


def make_preprocessor(name: str, *, k: int | str = "all") -> Pipeline:
    """Build a fold-local preprocessing pipeline.

    k counts encoded feature columns. The paper does not specify a cutoff,
    so the default keeps all columns. Set k to select by information gain.
    """
    name = _dataset_name(name)
    categorical = list(_CATEGORICAL.get(name, ()))
    numerical = [index for index in range(_SHAPES[name][1]) if index not in categorical]
    columns = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", MinMaxScaler()),
        ]), numerical),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
    return Pipeline([
        ("columns", columns),
        ("select", SelectKBest(score_func=_information_gain, k=k)),
    ])


if __name__ == "__main__":
    for name in DATASETS:
        X, _ = load_dataset(name)
        print(f"{name}: {X.shape[0]} samples, {X.shape[1]} features")
