
import argparse
from itertools import product
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from Source.baseline import C50Classifier, StandardKNN
from Source.data_loader import DATASETS, load_dataset, make_preprocessor
from Source.knn_model import KNNModel

K_VALUES = (1, 3, 5)


def cv_accuracy(name, estimator, X, y, cv) -> float:
    """Accuracy trung bình (%) qua các fold; tiền xử lý được fit riêng trong từng fold."""
    pipeline = make_pipeline(make_preprocessor(name), estimator)
    scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="accuracy", error_score="raise"
    )
    return float(scores.mean() * 100)


def best_knnmodel(name, X, y, cv, r_values, n_values, n_jobs):
    """Quét lưới (r, N); trả về (best_acc, r, N). Hòa thì lấy r nhỏ, rồi N nhỏ."""
    grid = list(product(r_values, n_values))
    accs = Parallel(n_jobs=n_jobs)(
        delayed(cv_accuracy)(name, KNNModel(r=r, N=n), X, y, cv) for r, n in grid
    )
    best = int(np.argmax(accs))  # argmax lấy phần tử đầu tiên khi hòa
    r, n = grid[best]
    return accs[best], r, n


def run(args) -> list[dict]:
    data_dir = Path(__file__).resolve().parents[1] / "Data"
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    r_values = range(args.r_min, args.r_max + 1)
    n_values = range(args.n_min, args.n_max + 1)

    rows = []
    for name in DATASETS:
        print(f"... {name}", flush=True)
        X, y = load_dataset(name, data_home=data_dir)
        best_acc, r, n = best_knnmodel(name, X, y, cv, r_values, n_values, args.jobs)
        row = {
            "Dataset": name.capitalize() if name != "aust" else "Aust",
            "C5.0": cv_accuracy(name, C50Classifier(), X, y, cv),
            "Best": best_acc,
            "r": r,
            "N": n,
        }
        for k in K_VALUES:
            row[f"K={k}"] = cv_accuracy(name, StandardKNN(n_neighbors=k), X, y, cv)
        rows.append(row)
    return rows


def format_table(rows: list[dict]) -> str:
    acc_cols = ["C5.0", "Best", "K=1", "K=3", "K=5"]
    average = {c: float(np.mean([row[c] for row in rows])) for c in acc_cols}
    header = ["Dataset", "C5.0", "Best Acc", "r", "N", "K=1", "K=3", "K=5"]
    lines = [" | ".join(f"{h:>10}" for h in header)]
    lines.append("-" * len(lines[0]))
    for row in rows:
        cells = [
            f"{row['Dataset']:>10}", f"{row['C5.0']:>10.2f}", f"{row['Best']:>10.2f}",
            f"{row['r']:>10}", f"{row['N']:>10}",
            *(f"{row[f'K={k}']:>10.2f}" for k in K_VALUES),
        ]
        lines.append(" | ".join(cells))
    lines.append("-" * len(lines[0]))
    lines.append(" | ".join([
        f"{'Average':>10}", f"{average['C5.0']:>10.2f}", f"{average['Best']:>10.2f}",
        f"{'':>10}", f"{'':>10}",
        *(f"{average[f'K={k}']:>10.2f}" for k in K_VALUES),
    ]))
    return "\n".join(lines)


def save_csv(rows: list[dict], path: str) -> None:
    import pandas as pd

    frame = pd.DataFrame(rows)
    mean_row = frame.drop(columns=["Dataset", "r", "N"]).mean()
    frame.loc[len(frame)] = {"Dataset": "Average", **mean_row.to_dict()}
    frame.round(2).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--r-min", type=int, default=1)
    parser.add_argument("--r-max", type=int, default=5)
    parser.add_argument("--n-min", type=int, default=1)
    parser.add_argument("--n-max", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=-1, help="-1 = dùng mọi core, 1 = tuần tự")
    parser.add_argument("--csv", default=None, help="đường dẫn file CSV để lưu kết quả")
    args = parser.parse_args()

    rows = run(args)
    print("\nTable 4. The influence of r and N to kNNModel (accuracy %)\n")
    print(format_table(rows))
    if args.csv:
        save_csv(rows, args.csv)
        print(f"\nĐã lưu {args.csv}")


if __name__ == "__main__":
    main()