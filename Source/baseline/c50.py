"""
Baseline 2: C5.0 Decision Tree Classifier
Theo bài báo "KNN Model-Based Approach in Classification" (Guo et al., 2003),
C5.0 (từ gói phần mềm Clementine / SPSS Modeler) là thuật toán cây quyết định
dựa trên Entropy và Information Gain của Ross Quinlan.

Trong Python, mô hình này được ánh xạ chuẩn thông qua DecisionTreeClassifier
với criterion='entropy' (hoặc thư viện pyc50 nếu có).
"""

from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier


class C50Classifier(BaseEstimator, ClassifierMixin):
    """
    C5.0 Decision Tree Baseline Classifier.
    
    Parameters
    ----------
    criterion : str, default='entropy'
        Độ đo thông tin phân chia nhánh (Entropy/Information Gain như C4.5/C5.0).
    max_depth : Optional[int], default=None
        Độ sâu tối đa của cây (mặc định None để cây phân nhánh tự do theo quy tắc dừng chuẩn).
    min_samples_split : int, default=2
        Số mẫu tối thiểu để phân chia 1 nút.
    random_state : Optional[int], default=42
        Seed ngẫu nhiên đảm bảo tính tái lập.
    """
    def __init__(
        self,
        criterion: str = 'entropy',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: Optional[int] = 42
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        
        self._model = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._model.fit(X, y)
        self.classes_ = self._model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)


def get_c50() -> C50Classifier:
    return C50Classifier()
