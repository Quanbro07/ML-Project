# Báo cáo Thực nghiệm Tái lập Bảng 3: kNN Model-Based Approach in Classification (Guo et al., 2003)

## 1. Điều chỉnh Mô hình & Tiền xử lý (Model & Preprocessing Modifications)

### 1.1. Hiệu chỉnh định nghĩa kích thước vùng lân cận $Num(d_i)$ trong `KNNModel`
* **Cơ sở từ bài báo (Section 3.2, trang 5):**
  > *"The representative $<Cls(d_i), Sim(d_i), Num(d_i), Rep(d_i)>$ respectively represents the class label of $d_i$, the lowest similarity to $d_i$ among the data tuples covered by $N_i$; **the number of data tuples covered by $N_i$** and a representation of $d_i$ itself."*
* **Vấn đề ở phiên bản cài đặt đầu tiên:**
  Khi thiết lập ngưỡng dung nạp lỗi $r = 1$, thuật toán mở rộng quả cầu cho phép bao phủ cả các điểm khác lớp (`pending_errors`) nằm lọt bên trong giữa các điểm cùng lớp. Tuy nhiên, biến `num` của `Representative` ban đầu chỉ đếm số điểm cùng nhãn (`committed_same_count`), bỏ qua các điểm khác nhãn bị bao phủ bên trong. Điều này khiến kích thước cụm bị hụt đi $r$ đơn vị, dẫn đến việc cắt tỉa theo ngưỡng $N > i$ (`rep.num <= self.N`) loại bỏ nhầm các cụm hợp lệ ở các mốc $N > 2, 3, 4, 5$.
* **Thay đổi trong code (`knn_model.py`):**
  Giữ nguyên tiêu chí tìm vùng lân cận lớn nhất theo số điểm cùng nhãn ở Bước 3–4, nhưng cập nhật thuộc tính lưu trữ `num` bằng tổng số điểm thực tế nằm trong vùng bao phủ:
  ```python
  representatives.append(
      Representative(
          cls=y[best_candidate_idx],
          sim=best_sim,
          num=len(best_covered_indices),  # Tổng số điểm bao phủ (Num(d_i))
          rep=X[best_candidate_idx].copy(),
          member_indices=best_covered_indices,
      )
  )
  ```

### 1.2. Thử nghiệm Tiền xử lý: `k="all"` vs. Feature Selection (`TOP_K` bằng Information Gain)
* **Giả thuyết về không gian đặc trưng:**
  Hai bộ dữ liệu `Heart` và `Aust` chứa các thuộc tính định danh (Nominal). Khi áp dụng `OneHotEncoder`, số chiều không gian bị mở rộng đáng kể (`Heart` từ 13 lên 20 chiều; `Aust` từ 14 lên 42 chiều). Trong không gian khoảng cách Euclidean, việc tăng chiều đột ngột từ các biến giả (dummy variables) làm loãng mật độ điểm (curse of dimensionality) và tăng trọng số sai khác mỗi khi hai mẫu khác giá trị nominal.
* **Thực hiện:**
  So sánh giữa việc giữ toàn bộ đặc trưng sau encode (`k="all"`) và kích hoạt bộ lọc `SelectKBest` dựa trên **Information Gain** (đúng với mô tả ở Section 4 của bài báo) với cấu hình `TOP_K` tinh gọn cho từng tập dữ liệu (`glass: 7` hoặc `all`, `iris: 3`, `heart: 11`, `wine: 10`, `diabetes: 6`, `aust: 12`).

---

## 2. Giả thuyết Dự đoán Trước Khi Chạy Thực nghiệm

Trước khi tiến hành chạy thực nghiệm quét ngưỡng cắt tỉa $N > i$ ($i \in \{1, 2, 3, 4, 5\}$ với $r = 1$), nhóm đặt ra hai dự đoán:
1. **Về tỷ lệ nén (Reduction Rate):** Khi $i$ tăng lên, ngưỡng yêu cầu số lượng phần tử tối thiểu trong mỗi đại diện ($Num(d_i) \ge i + 1$) trở nên khắt khe hơn. Các cụm nhỏ lẻ bị loại bỏ nhiều hơn, do đó **tỷ lệ nén chắc chắn sẽ tăng dần theo $i$**.
2. **Về độ chính xác (Classification Accuracy):** Ban đầu nhóm dự đoán **độ chính xác có thể sẽ tăng nhẹ hoặc giữ ổn định**, với kỳ vọng việc xóa bỏ các đại diện có kích thước nhỏ sẽ giúp mô hình loại bỏ nhiễu (outliers) và các điểm ngoại lai ở vùng biên.

---

## 3. Kết quả Thực nghiệm & Phân tích Sai lệch

### 3.1. Những điểm trùng khớp với bài báo gốc
* **Xu hướng nén dữ liệu (Table 3a & 3b):** Số lượng đại diện giảm đều và tỷ lệ nén tăng tịnh tiến khi $i$ chạy từ $1 \to 5$. Tỷ lệ nén trung bình đạt **90.95% – 96.00%**, bám rất sát mức **89.87% – 93.70%** được công bố trong Table 3 của bài báo.
* **Hành vi sau khi sửa `num = len(best_covered_indices)`:** Số lượng đại diện tại các mốc $N > 2 \dots N > 5$ tăng lên hợp lý (ví dụ `Diabetes` ở $N>2$ tăng từ $56.20$ lên $64.40$, `Glass` từ $16.80$ lên $18.20$), phản ánh đúng cơ chế giữ lại các cụm có 2 điểm cùng lớp + 1 điểm lỗi.
* **Độ chính xác ở mốc chuẩn ($N > 1$):** Sau khi bật Feature Selection bằng Information Gain, các bộ dữ liệu tiệm cận sát kết quả Table 2 của bài báo: `Iris` đạt **96.00%** (khớp 100%), `Heart` đạt **80.74%** (so với 80.37%), `Diabetes` đạt **74.60%** (so với 73.59%), `Aust` đạt **84.64% – 85.22%** (so với 85.65%).

### 3.2. Phân tích hiện tượng Accuracy giảm khi $i$ tăng (Trái ngược dự đoán ban đầu)
Thực tế chạy nghiệm cho thấy khi tăng $i$ từ $1 \to 5$, độ chính xác trung bình **giảm dần từ 83.21% xuống 80.40%** (mức giảm ~2.8% – 3%, đặc biệt rõ ở các tập đa lớp như `Glass` giảm từ 67.75% xuống 60.72%). Hiện tượng này được lý giải qua 3 khía cạnh:

1. **Bản chất của Pruning $N > i$ (Underfitting vùng biên thay vì chỉ lọc nhiễu):**
   Ở $N > 1$ (xóa cụm 1 điểm), mô hình loại bỏ các điểm nhiễu cô lập thực sự nên accuracy đạt đỉnh. Tuy nhiên, khi đẩy lên $N > 4$ hoặc $N > 5$ (ép mỗi đại diện phải phủ $\ge 5\text{–}6$ điểm), thuật toán xóa luôn cả các **cụm nhỏ hợp lệ nằm dọc theo ranh giới quyết định (decision boundary)** hoặc các nhóm thiểu số (như các lớp hiếm trong `Glass`). Khi mất các đại diện vùng biên, các điểm test nằm gần ranh giới bị ép phân loại theo khoảng cách tới biên của các quả cầu lõi rất lớn ở xa ($d(d_i, d_t) - Sim(d_i)$), làm mất độ phân giải phân lớp.

2. **Tác động của Không gian Đầu vào (Euclidean Space) vs. Không gian Tiềm ẩn (Latent / Manifold):**
   * *Hiệu chỉnh về mặt lý thuyết:* Giả thuyết đa tạp (Manifold Hypothesis) chỉ ra rằng dữ liệu chiều cao thường nằm trên một đa tạp có số chiều nội tại thấp (low-dimensional manifold) chứa các đặc trưng bản chất (latent patterns).
   * *Chuyện gì xảy ra khi cắt tỉa trên không gian đầu vào thô (Euclidean Space):* Ở đây, `kNNModel` hoạt động trực tiếp trên không gian đặc trưng gốc (sau Min-Max và One-Hot) bằng khoảng cách Euclidean đẳng hướng, chứ **không phải** trên một không gian biểu diễn tiềm ẩn (latent space) đã được học để gom cụm theo ngữ nghĩa. Trên không gian đầu vào thô, hình dạng các lớp thường bị kéo giãn, uốn lượn hoặc đứt gãy thành nhiều cụm nhỏ cục bộ để uốn theo đường cong của đa tạp. Do đó, việc xóa các cụm nhỏ ($N > i$ lớn) trên không gian Euclidean **gây tổn hại lớn hơn** vì vô tình cắt đứt cấu trúc hình học cục bộ của đa tạp; ngược lại, nếu dữ liệu đã được ánh xạ vào một **latent space** tốt (nơi các điểm cùng lớp đã được nén chặt thành một khối cầu tập trung), việc loại bỏ các cụm nhỏ lẻ sẽ ít làm tổn hại tới ranh giới phân lớp hơn.

3. **Các yếu tố sai khác về thiết lập thực nghiệm so với bài báo năm 2003:**
   * **Phân chia 5-fold CV & kích thước mẫu nhỏ:** Bài báo không công bố `random_state` chia fold. Với các tập nhỏ như `Iris` (30 mẫu test/fold) hay `Wine` (~35 mẫu test/fold), sai khác chỉ **1 mẫu dự đoán** đã làm accuracy dao động **2.8% – 3.3%**.
   * **Công cụ tiền xử lý & Data Leakage:** Bài báo gốc sử dụng phần mềm *Clementine* (năm 2003) và thực hiện chuẩn hóa + chọn đặc trưng trên toàn bộ tập dữ liệu trước khi chia fold (*"six datasets were pre-processed... before conducting the classification"*). Trong khi đó, thực nghiệm tái lập tuân thủ chuẩn chống rò rỉ dữ liệu hiện đại (`Pipeline` nằm trong `cross_validate`), dẫn đến độ chính xác thực tế thấp hơn khoảng 0.5% – 1% so với con số công bố.
   * **Độ đo khoảng cách cho biến Nominal:** `OneHotEncoder` tạo ra khoảng cách $\sqrt{2}$ giữa hai giá trị khác nhau của cùng một thuộc tính nominal, thay vì khoảng cách chuẩn hóa $1$ (Overlap metric) thường dùng trong các hệ thống khai phá dữ liệu truyền thống.

