"""
kNNModel: A Model-Based Approach in Classification
Dựa trên bài báo nghiên cứu:
"KNN Model-Based Approach in Classification"
(Gongde Guo, Hui Wang, David Bell, Yaxin Bi, Kieran Greer, 2003)

Mô hình nén dữ liệu huấn luyện thành một tập các đại diện lân cận hình cầu
dạng bộ 4 thành phần: <Cls, Sim, Num, Rep>
hỗ trợ 2 kỹ thuật cắt tỉa (Pruning 1: N, Pruning 2: r) và 3 luật phân loại.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


@dataclass
class Representative:
    """
    Cấu trúc lưu trữ một đại diện hình cầu trong mô hình M:
    <Cls(d_i), Sim(d_i), Num(d_i), Rep(d_i)>
    
    Attributes
    ----------
    cls : Any
        Nhãn lớp của tâm đại diện.
    sim : float
        Bán kính hình cầu (khoảng cách Euclidean tới điểm xa nhất trong cụm lân cận).
    num : int
        Tổng số điểm dữ liệu được bao phủ bởi hình cầu này.
    rep : np.ndarray
        Vector đặc trưng (tọa độ) của tâm đại diện d_i.
    member_indices : List[int]
        Danh sách chỉ số các điểm huấn luyện thuộc cụm này (dùng cho Post-pruning).
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
        Độ chấp nhận lỗi (Error-tolerant degree - Pruning phương pháp 2).
        Số lượng điểm khác nhãn tối đa được phép 'nuốt' vào trong vùng lân cận
        khi mở rộng hình cầu. Mặc định r=0 (vùng lân cận thuần nhất 100%).
        
    N : int, default=0
        Ngưỡng cắt tỉa sau huấn luyện (Post-pruning threshold - Pruning phương pháp 1).
        Quy ước N > i trong bài báo: loại bỏ các đại diện có Num <= N, xóa các điểm
        thuộc các đại diện đó khỏi tập huấn luyện, sau đó huấn luyện lại 1 lượt duy nhất.
        Mặc định N=0 (không cắt tỉa).
        
    metric : str, default='euclidean'
        Độ đo khoảng cách, mặc định là khoảng cách Euclidean.
    """
    def __init__(
        self,
        r: int = 0,
        N: int = 0,
        metric: str = 'euclidean'
    ):
        self.r = r
        self.N = N
        self.metric = metric

    def _build_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        r: int
    ) -> List[Representative]:
        """
        Thuật toán 6 bước xây dựng mô hình đại diện M từ tập dữ liệu.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        r : int, độ dung nạp lỗi
        
        Returns
        -------
        representatives : List[Representative]
            Tập hợp các bộ đại diện tạo thành mô hình M.
        """
        n_samples = len(X)
        if n_samples == 0:
            return []

        # Bước 1: Tạo ma trận khoảng cách O(n^2)
        dist_matrix = cdist(X, X, metric=self.metric)

        # Bước 2: Khởi tạo trạng thái 'ungrouped' (False = ungrouped, True = grouped)
        grouped = np.zeros(n_samples, dtype=bool)
        representatives: List[Representative] = []

        # Lặp Bước 3 và Bước 4 cho đến khi tất cả điểm đều 'grouped'
        while not np.all(grouped):
            ungrouped_indices = np.where(~grouped)[0]
            
            best_candidate_idx = None
            best_sim = float('inf')
            best_num = -1
            best_covered_indices: List[int] = []

            # Bước 3: Tìm lân cận cục bộ lớn nhất (Largest Local Neighbourhood)
            # cho từng điểm d_i đang ungrouped
            for i in ungrouped_indices:
                target_cls = y[i]
                
                # Khoảng cách từ d_i tới tất cả các điểm ungrouped khác
                dists_to_ungrouped = dist_matrix[i, ungrouped_indices]
                sorted_order = np.argsort(dists_to_ungrouped)
                sorted_ungrouped = ungrouped_indices[sorted_order]
                sorted_dists = dists_to_ungrouped[sorted_order]

                # Mở rộng vùng lân cận quanh d_i
                current_errors = 0
                covered_for_i: List[int] = []
                farthest_dist = 0.0

                for point_idx, d_val in zip(sorted_ungrouped, sorted_dists):
                    point_cls = y[point_idx]
                    
                    if point_cls != target_cls:
                        if current_errors < r:
                            # Chấp nhận lỗi theo tham số r
                            current_errors += 1
                            covered_for_i.append(int(point_idx))
                            farthest_dist = float(d_val)
                        else:
                            # Đã đạt giới hạn lỗi r, dừng mở rộng hình cầu
                            break
                    else:
                        covered_for_i.append(int(point_idx))
                        farthest_dist = float(d_val)

                num_covered = len(covered_for_i)

                # Bước 4: Chọn đại diện toàn cục (Largest Global Neighbourhood)
                # Tiêu chí: max(Num), nếu hòa (tie-breaking) thì min(Sim)
                is_better = False
                if num_covered > best_num:
                    is_better = True
                elif num_covered == best_num:
                    if farthest_dist < best_sim:
                        is_better = True

                if is_better:
                    best_candidate_idx = i
                    best_num = num_covered
                    best_sim = farthest_dist
                    best_covered_indices = covered_for_i

            # Tạo bộ đại diện 4 thành phần <Cls, Sim, Num, Rep>
            rep_obj = Representative(
                cls=y[best_candidate_idx],
                sim=best_sim,
                num=best_num,
                rep=X[best_candidate_idx].copy(),
                member_indices=best_covered_indices
            )
            representatives.append(rep_obj)

            # Đổi trạng thái các điểm nằm trong vùng phủ thành 'grouped'
            grouped[best_covered_indices] = True

        return representatives

    def fit(self, X: Any, y: Any):
        """
        Huấn luyện mô hình kNNModel trên tập dữ liệu X, y.
        Tự động áp dụng Pruning 1 (Post-pruning & Retraining) nếu N > 0.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        
        self.classes_ = np.unique(y_arr)
        
        # Lượt 1: Huấn luyện mô hình ban đầu với tham số r (Pruning 2)
        initial_reps = self._build_model(X_arr, y_arr, r=self.r)

        # Kiểm tra Pruning 1 (Post-pruning & Retraining với tham số N)
        # N > i nghĩa là loại bỏ các đại diện có Num <= N
        if self.N > 0:
            weak_points_mask = np.zeros(len(X_arr), dtype=bool)
            has_pruned = False

            for rep in initial_reps:
                if rep.num <= self.N:
                    has_pruned = True
                    weak_points_mask[rep.member_indices] = True

            if has_pruned:
                # Xóa các điểm thuộc các đại diện yếu để tạo D'
                keep_mask = ~weak_points_mask
                X_retrain = X_arr[keep_mask]
                y_retrain = y_arr[keep_mask]

                # Nếu còn dữ liệu sau khi lọc, chạy lại đúng 1 lượt
                if len(X_retrain) > 0 and len(np.unique(y_retrain)) > 0:
                    self.representatives_ = self._build_model(X_retrain, y_retrain, r=self.r)
                else:
                    self.representatives_ = initial_reps
            else:
                self.representatives_ = initial_reps
        else:
            self.representatives_ = initial_reps

        return self

    def predict_single(self, x_t: np.ndarray) -> Any:
        """
        Dự đoán nhãn cho 1 điểm dữ liệu mới x_t dựa trên 3 trường hợp của bài báo:
        - TH1: Chỉ nằm trong 1 vùng đại diện -> gán nhãn của đại diện đó.
        - TH2: Nằm trong vùng giao của >= 2 đại diện -> gán nhãn của đại diện có Num lớn nhất.
        - TH3: Nằm ngoài tất cả -> tính khoảng cách tới biên dt = d(x_t, Rep) - Sim,
               gán nhãn của đại diện có dt nhỏ nhất (1-NN tới biên).
        """
        if not self.representatives_:
            raise ValueError("Mô hình chưa có đại diện nào được huấn luyện!")

        # Tính khoảng cách Euclidean từ x_t tới tất cả các tâm đại diện
        rep_centers = np.array([rep.rep for rep in self.representatives_])
        dists = np.linalg.norm(rep_centers - x_t, axis=1)

        # Tìm các đại diện bao phủ x_t: d(x_t, Rep_j) <= Sim_j
        covering_reps: List[Tuple[Representative, float]] = []
        for rep, d_val in zip(self.representatives_, dists):
            if d_val <= rep.sim:
                covering_reps.append((rep, d_val))

        # TH1: dt chỉ nằm trong 1 vùng đại diện
        if len(covering_reps) == 1:
            return covering_reps[0][0].cls

        # TH2: dt nằm trong vùng giao của >= 2 đại diện
        elif len(covering_reps) > 1:
            # Kiểm tra nếu tất cả đại diện bao phủ đều cùng 1 nhãn
            covering_classes = [item[0].cls for item in covering_reps]
            if len(set(covering_classes)) == 1:
                return covering_classes[0]

            # Khác nhãn: chọn đại diện có Num lớn nhất
            # Quy tắc hòa: nếu hòa Num thì chọn Sim nhỏ nhất
            best_rep = max(
                covering_reps,
                key=lambda item: (item[0].num, -item[0].sim)
            )[0]
            return best_rep.cls

        # TH3: dt nằm ngoài tất cả các vùng đại diện
        else:
            # dt(d_t, d_i) = d(d_t, Rep(d_i)) - Sim(d_i)
            boundary_distances = dists - np.array([rep.sim for rep in self.representatives_])
            
            # Chọn đại diện có khoảng cách tới biên nhỏ nhất
            # Quy tắc hòa: nếu cùng dt thì chọn đại diện có Num lớn hơn
            min_boundary_dist = np.min(boundary_distances)
            candidate_indices = np.where(np.isclose(boundary_distances, min_boundary_dist, atol=1e-8))[0]
            
            if len(candidate_indices) == 1:
                return self.representatives_[candidate_indices[0]].cls
            else:
                best_idx = max(
                    candidate_indices,
                    key=lambda idx: (self.representatives_[idx].num, -self.representatives_[idx].sim)
                )
                return self.representatives_[best_idx].cls

    def predict(self, X: Any) -> np.ndarray:
        """
        Dự đoán nhãn cho tập mẫu kiểm tra X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        
        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
        """
        check_is_fitted(self, attributes=["representatives_", "classes_"])
        X_arr = np.asarray(X, dtype=np.float64)
        predictions = [self.predict_single(x_t) for x_t in X_arr]
        return np.array(predictions)
