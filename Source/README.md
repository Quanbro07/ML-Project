# KNN Model-Based Classification Project

Dự án cài đặt và thực nghiệm thuật toán từ bài báo:
> **"KNN Model-Based Approach in Classification"** *(Gongde Guo, Hui Wang, David Bell, Yaxin Bi, Kieran Greer, 2003)*.

## 1. Cấu trúc thư mục

```
Source/
├── baseline/
│   ├── standard_knn.py   # Baseline Standard k-NN (k=1, 3, 5, metric='euclidean')
│   ├── c50.py            # Baseline C5.0 (DecisionTreeClassifier với criterion='entropy')
│   └── __init__.py       # Module export tiện ích
├── knn_model.py          # Lớp KNNModel cài đặt thuật toán đề xuất của bài báo
├── data_loader.py        # Nạp sáu bộ dữ liệu và tạo pipeline tiền xử lý
├── requirements.txt      # Thư viện phụ thuộc
└── README.md
```

## 2. Các thành phần chính của `KNNModel`

* **Cấu trúc bộ đại diện:** $\langle Cls, Sim, Num, Rep \rangle$
  * `Cls`: Nhãn lớp của đại diện.
  * `Sim`: Bán kính hình cầu (khoảng cách Euclidean tới điểm xa nhất trong cụm).
  * `Num`: Số lượng điểm dữ liệu được bao phủ.
  * `Rep`: Vector đặc trưng tọa độ tâm đại diện.

* **Hai cơ chế Pruning:**
  * `r` (*Error-tolerant degree* - Pruning 2): Dung nạp tối đa $r$ điểm khác nhãn trong quá trình mở rộng lân cận hình cầu khi huấn luyện.
  * `N` (*Post-pruning & Retraining* - Pruning 1): Loại bỏ các đại diện có $Num \le N$, xóa các điểm dữ liệu thuộc các đại diện đó khỏi tập huấn luyện và train lại 1 lượt duy nhất trên tập dữ liệu còn lại.

* **Ba luật phân loại mẫu mới:**
  * **TH1:** Điểm nằm trong duy nhất 1 hình cầu đại diện $\to$ gán nhãn đại diện đó.
  * **TH2:** Điểm nằm trong vùng giao của $\ge 2$ đại diện khác nhãn $\to$ chọn đại diện có $\max Num$ (phá hòa bằng $\min Sim$).
  * **TH3:** Điểm nằm ngoài tất cả các hình cầu $\to$ tính $dt(x, Rep) = d(x, Rep) - Sim$, chọn đại diện có bề mặt gần nhất ($\min dt$).

## 3. Cách sử dụng

Chạy ví dụ dưới đây từ thư mục `Source/`. Tên bộ dữ liệu hợp lệ gồm
`glass`, `iris`, `heart`, `wine`, `diabetes` và `aust`.
Từ thư mục gốc, chạy `python Source/data_loader.py` để tải các bộ dữ liệu UCI
còn thiếu bằng `ucimlrepo` và lưu vào `Data/`. Những lần chạy sau đọc CSV
tại chỗ. Diabetes dùng tệp `Data/Diabetes.csv` có sẵn.

```python
from data_loader import load_dataset, make_preprocessor
from knn_model import KNNModel
from baseline import StandardKNN, C50Classifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline

name = "iris"
X, y = load_dataset(name)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "kNNModel": KNNModel(r=1, N=1),
    **{f"{k}-NN": StandardKNN(n_neighbors=k) for k in (1, 3, 5)},
    "C5.0 baseline": C50Classifier(),
}
for label, model in models.items():
    pipeline = make_pipeline(make_preprocessor(name, k=3), model)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print(f"{label} 5-fold accuracy = {scores.mean():.4f}")
```

Pipeline điền giá trị thiếu, chuẩn hóa min–max cho cột số và mã hóa one-hot
cho cột danh nghĩa. Mỗi bước được fit trên phần huấn luyện của từng fold.
Bài báo không nêu số đặc trưng giữ lại. Mặc định `k="all"` giữ toàn bộ cột,
còn `k=3` trong ví dụ chọn ba cột sau mã hóa bằng information gain.
Truyền `data_home` nếu muốn đọc hoặc lưu CSV ở thư mục khác.
