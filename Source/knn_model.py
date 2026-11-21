"""
kNNModel: A Model-Based Approach in Classification
Dựa trên bài báo: "KNN Model-Based Approach in Classification"
(Gongde Guo, Hui Wang, David Bell, Yaxin Bi, Kieran Greer, 2003)
"""

from dataclasses import dataclass
from typing import Any, List, Tuple
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


@dataclass
class Representative:
    """
    Đại diện hình cầu trong mô hình M: <Cls(d_i), Sim(d_i), Num(d_i), Rep(d_i)>
    """
    cls: Any
    sim: float
    num: int
    rep: np.ndarray
    member_indices: List[int]


class KNNModel(BaseEstimator, ClassifierMixin):
    """
    KNN Model-Based Classifier (Guo et al., 2003).

    Parameters
    ----------
    r : int, default=0
        Error-tolerant degree (Pruning 2): số điểm khác nhãn tối đa được phép
        nằm lọt trong vùng lân cận khi mở rộng hình cầu.
    N : int, default=0
        Post-pruning threshold (Pruning 1): loại bỏ các đại diện có Num <= N
        cùng các điểm thuộc đại diện đó, rồi huấn luyện lại 1 lượt.
        (Theo quy ước N > i trong bài báo: truyền N=1 tương ứng N > 1).
    metric : str, default='euclidean'
        Độ đo khoảng cách truyền cho scipy.spatial.distance.cdist.
    """

    def __init__(self, r: int = 0, N: int = 0, metric: str = "euclidean"):
        self.r = r
        self.N = N
        self.metric = metric

    def _build_model(
        self, X: np.ndarray, y: np.ndarray, r: int
    ) -> List[Representative]:
        """Xây dựng tập đại diện M theo thuật toán 6 bước của Guo et al. (2003)."""
        n_samples = len(X)
        if n_samples == 0:
            return []

        # Bước 1: Tạo ma trận khoảng cách và sắp xếp lân cận 1 lần duy nhất
        dist_matrix = cdist(X, X, metric=self.metric)
        sorted_all_indices = np.argsort(dist_matrix, axis=1)

        # Bước 2: Khởi tạo trạng thái 'ungrouped'
        grouped = np.zeros(n_samples, dtype=bool)
        representatives: List[Representative] = []

        # Bước 3–5: Lặp cho đến khi toàn bộ điểm đều được gán nhãn 'grouped'
        while not np.all(grouped):
            ungrouped_indices = np.where(~grouped)[0]
            best_candidate_idx = None
            best_sim = float("inf")
            best_num = -1
            best_covered_indices: List[int] = []

            for i in ungrouped_indices:
                target_cls = y[i]
                neighbors = sorted_all_indices[i]
                dists = dist_matrix[i, neighbors]

                current_errors = 0
                committed_covered: List[int] = [int(i)]
                committed_same_count = 1
                committed_farthest_dist = 0.0
                pending_errors: List[int] = []

                for point_idx, d_val in zip(neighbors, dists):
                    if point_idx == i:
                        continue

                    if y[point_idx] != target_cls:
                        # Mọi điểm khác lớp (kể cả đã grouped) đều tính vào ngân sách lỗi r
                        if current_errors < r:
                            current_errors += 1
                            if not grouped[point_idx]:
                                pending_errors.append(int(point_idx))
                        else:
                            break
                    else:
                        if not grouped[point_idx]:
                            # Chỉ gộp các điểm lỗi nằm lọt bên trong giữa các điểm cùng lớp
                            if pending_errors:
                                committed_covered.extend(pending_errors)
                                pending_errors = []
                            committed_covered.append(int(point_idx))
                            committed_same_count += 1
                            committed_farthest_dist = float(d_val)

                # Bước 4: Chọn largest global neighbourhood (max Num, hòa thì min Sim)
                if (committed_same_count > best_num) or (
                    committed_same_count == best_num
                    and committed_farthest_dist < best_sim
                ):
                    best_candidate_idx = i
                    best_num = committed_same_count
                    best_sim = committed_farthest_dist
                    best_covered_indices = committed_covered

            representatives.append(
                Representative(
                    cls=y[best_candidate_idx],
                    sim=best_sim,
                    num=best_num,
                    rep=X[best_candidate_idx].copy(),
                    member_indices=best_covered_indices,
                )
            )
            grouped[best_covered_indices] = True

        return representatives

    def fit(self, X: Any, y: Any):
        """Huấn luyện kNNModel, áp dụng Pruning 1 (N) và Pruning 2 (r)."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        self.classes_ = np.unique(y_arr)
        self.n_train_samples_ = len(X_arr)

        # Lượt 1: xây dựng mô hình ban đầu với độ dung nạp lỗi r
        initial_reps = self._build_model(X_arr, y_arr, r=self.r)

        # Pruning 1: loại bỏ đại diện có Num <= N cùng các điểm thuộc đại diện đó rồi train lại
        if self.N > 0:
            weak_points_mask = np.zeros(len(X_arr), dtype=bool)
            has_pruned = False

            for rep in initial_reps:
                if rep.num <= self.N:
                    has_pruned = True
                    weak_points_mask[rep.member_indices] = True

            if has_pruned:
                keep_mask = ~weak_points_mask
                X_retrain = X_arr[keep_mask]
                y_retrain = y_arr[keep_mask]

                if len(X_retrain) > 0:
                    self.representatives_ = self._build_model(
                        X_retrain, y_retrain, r=self.r
                    )
                else:
                    self.representatives_ = initial_reps
            else:
                self.representatives_ = initial_reps
        else:
            self.representatives_ = initial_reps

        # Lưu thông tin nén và mảng tâm/bán kính để dự đoán nhanh
        self.n_representatives_ = len(self.representatives_)
        self.reduction_rate_ = (
            (1.0 - self.n_representatives_ / self.n_train_samples_) * 100.0
            if self.n_train_samples_ > 0
            else 0.0
        )
        self._rep_centers_ = np.array(
            [rep.rep for rep in self.representatives_], dtype=np.float64
        )
        self._rep_sims_ = np.array(
            [rep.sim for rep in self.representatives_], dtype=np.float64
        )
        return self

    def _predict_from_dists(self, dists: np.ndarray) -> Any:
        """Phân loại 1 điểm từ vector khoảng cách tới các tâm đại diện theo 3 luật của bài báo."""
        covering_reps: List[Tuple[Representative, float]] = [
            (rep, float(d_val))
            for rep, d_val in zip(self.representatives_, dists)
            if d_val <= rep.sim
        ]

        # Luật 1: Chỉ nằm trong 1 vùng đại diện
        if len(covering_reps) == 1:
            return covering_reps[0][0].cls

        # Luật 2: Nằm trong vùng giao của >= 2 đại diện -> chọn Num lớn nhất (hòa thì Sim nhỏ nhất)
        if len(covering_reps) > 1:
            covering_classes = {item[0].cls for item in covering_reps}
            if len(covering_classes) == 1:
                return covering_reps[0][0].cls
            best_rep = max(
                covering_reps, key=lambda item: (item[0].num, -item[0].sim)
            )[0]
            return best_rep.cls

        # Luật 3: Nằm ngoài tất cả -> chọn đại diện có khoảng cách tới biên d(x_t, Rep) - Sim nhỏ nhất
        boundary_distances = dists - self._rep_sims_
        min_boundary_dist = np.min(boundary_distances)
        candidate_indices = np.where(
            np.isclose(boundary_distances, min_boundary_dist, atol=1e-8)
        )[0]

        if len(candidate_indices) == 1:
            return self.representatives_[candidate_indices[0]].cls

        best_idx = max(
            candidate_indices,
            key=lambda idx: (
                self.representatives_[idx].num,
                -self.representatives_[idx].sim,
            ),
        )
        return self.representatives_[best_idx].cls

    def predict_single(self, x_t: np.ndarray) -> Any:
        """Dự đoán nhãn cho 1 mẫu dữ liệu."""
        check_is_fitted(self, attributes=["representatives_", "classes_"])
        if not self.representatives_:
            raise ValueError("Mô hình chưa có đại diện nào được huấn luyện!")
        x_arr = np.asarray(x_t, dtype=np.float64).reshape(1, -1)
        dists = cdist(x_arr, self._rep_centers_, metric=self.metric)[0]
        return self._predict_from_dists(dists)

    def predict(self, X: Any) -> np.ndarray:
        """Dự đoán nhãn cho tập mẫu kiểm tra X."""
        check_is_fitted(self, attributes=["representatives_", "classes_"])
        if not self.representatives_:
            raise ValueError("Mô hình chưa có đại diện nào được huấn luyện!")
        X_arr = np.asarray(X, dtype=np.float64)
        dist_matrix = cdist(X_arr, self._rep_centers_, metric=self.metric)
        return np.array([self._predict_from_dists(dists) for dists in dist_matrix])


# Alias tương thích nếu notebook đang gọi tên KNNModelFixed
KNNModelFixed = KNNModel