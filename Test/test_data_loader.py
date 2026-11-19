from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from Source.baseline import C50Classifier, StandardKNN
from Source.data_loader import DATASETS, load_dataset, make_preprocessor
from Source.knn_model import KNNModel


def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "Data"
    csv_files = sorted(data_dir.glob("*.csv"))
    missing = set(DATASETS) - {path.stem.lower() for path in csv_files}
    if missing:
        raise FileNotFoundError(f"Thiếu CSV trong {data_dir}: {', '.join(sorted(missing))}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        **{f"{k}-NN": StandardKNN(n_neighbors=k) for k in (1, 3, 5)},
        "C5.0 baseline": C50Classifier(),
        "KNNModel": KNNModel(r=1, N=1),
    }

    print("Accuracy trung bình qua 5 fold")
    print(f"{'Dataset':<15}{'Mẫu':>6}{'Cột':>6}", end="")
    print("".join(f"{label:>16}" for label in models))
    for path in csv_files:
        name = path.stem.lower()
        X, y = load_dataset(name, data_home=data_dir)
        accuracies = []
        for estimator in models.values():
            pipeline = make_pipeline(make_preprocessor(name), estimator)
            scores = cross_val_score(
                pipeline, X, y, cv=cv, scoring="accuracy", error_score="raise"
            )
            accuracies.append(scores.mean())
        print(
            f"{path.name:<15}{X.shape[0]:>6}{X.shape[1]:>6}"
            + "".join(f"{accuracy:>16.2%}" for accuracy in accuracies)
        )


if __name__ == "__main__":
    main()
