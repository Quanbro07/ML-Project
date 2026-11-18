"""
Baseline 1: Standard k-Nearest Neighbors (k-NN)
Theo bài báo "KNN Model-Based Approach in Classification" (Guo et al., 2003),
mô hình standard k-NN sử dụng khoảng cách Euclidean với các giá trị k thông dụng như k = 1, 3, 5.
"""

from typing import Union
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier


class StandardKNN(BaseEstimator, ClassifierMixin):
    """
    Standard k-NN Classifier wrapper sử dụng scikit-learn.
    
    Parameters
    ----------
    n_neighbors : int, default=3
        Số lượng lân cận k (thường đánh giá k=1, k=3, k=5 theo bài báo).
    metric : str, default='euclidean'
        Độ đo khoảng cách, mặc định là Euclidean.
    """
    def __init__(self, n_neighbors: int = 3, metric: str = 'euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self._model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._model.fit(X, y)
        self.classes_ = self._model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)


def get_standard_knn(k: int = 3) -> StandardKNN:
    """Hàm tiện ích tạo nhanh mô hình k-NN với tham số k."""
    return StandardKNN(n_neighbors=k)
