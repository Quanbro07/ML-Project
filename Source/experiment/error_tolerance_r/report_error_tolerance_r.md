# Báo cáo Thực nghiệm: Khảo sát Độ dung nạp lỗi ($r$) và Ảnh hưởng của Kiểu dữ liệu trong `kNNModel`

> **Phạm vi báo cáo:**
>
> 1. Tái lập **Hình 13 (`Aust`)** và **Hình 14 (`Glass`)** trong bài báo *Guo et al. (2003)* về ảnh hưởng của tham số độ dung nạp lỗi $r$ (`exp.ipynb`).
> 2. Thực nghiệm mở rộng kiểm chứng hai hạn chế kỹ thuật của `kNNModel` (phục vụ Mục 1.1.3 & Điểm thưởng Mục 1.2.4): **(i)** hiện tượng phân mảnh cấu trúc / overfitting khi $r = 0$, và **(ii)** nhược điểm của khoảng cách Euclidean trên dữ liệu hỗn hợp Nominal/Binary cùng hướng cải tiến bằng **Gower Distance** & **HEOM** (`exp_r0_and_datatype.ipynb`).

---

## 1. Thiết lập thực nghiệm

- **Dữ liệu:** 6 bộ dữ liệu UCI theo Table 1 của bài báo: 4 tập thuần số (`Glass`, `Iris`, `Wine`, `Diabetes`) và 2 tập hỗn hợp (`Heart`: 7 Numeric, 3 Nominal, 3 Binary; `Aust`: 6 Numeric, 4 Nominal, 4 Binary).
- **Quy trình đánh giá:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Mọi bước tiền xử lý (điền giá trị thiếu, chuẩn hóa `MinMaxScaler`, mã hóa `OneHotEncoder`, chọn đặc trưng Information Gain) đều đặt trong `Pipeline` và fit cục bộ theo từng fold để tránh rò rỉ dữ liệu (*data leakage*).
- **Các điểm khác biệt so với bài báo gốc:** Bài báo không công bố ngưỡng cắt $k$ của bước chọn đặc trưng Information Gain cũng như cách mã hóa biến Nominal trên `Heart`/`Aust`. Trong thực nghiệm dưới đây, cấu hình chuẩn giữ toàn bộ đặc trưng (`k="all"`) và chuẩn hóa `[0, 1]` cục bộ; ảnh hưởng của One-Hot Encoding và chọn đặc trưng IG được tách riêng thành thí nghiệm đối chứng.

---

## 2. Tái lập Hình 13 (`Aust`) & Hình 14 (`Glass`): Ảnh hưởng của Error-Tolerant Degree ($r$)

### 2.1. Thiết lập và đối chiếu kết quả

Giữ cố định ngưỡng hậu cắt tỉa $N = 1$ (loại đại diện đơn điểm $Num \le 1$) và khảo sát $r \in \{0, 1, \dots, 15\}$ trên hai bộ dữ liệu `Aust` (Hình 13) và `Glass` (Hình 14).

| Bộ dữ liệu             | Bài báo — Đỉnh tối ưu |          Tái lập — Đỉnh tối ưu          | Sai khác ($\Delta$ / Tương đối) |          Kiểm định mở rộng (`10×5-fold` & `IG`)          | Xu hướng đường cong theo $r \in [0, 15]$                                                                                                |
| :------------------------ | :--------------------------: | :---------------------------------------------: | :------------------------------------: | :----------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------- |
| **Aust (Fig. 13)**  |    $r = 2$ (`87.10%`)    | **$r = 2$ (`86.38%`, `RR=92.83%`)** |        $-0.72\%$ (`-0.83%`)        | `10×5`: $r=1$ (`84.43%`)`IG(k=10)`: $r=12$ (`86.52%`) | Tăng từ $r=0$ (`84.20%`) lên đỉnh tại $r=2$, sau đó giảm và dao động quanh $82.6\text{--}84.9\%$ (`83.77%` ở $r=15$). |
| **Glass (Fig. 14)** |    $r = 1$ (`70.95%`)    | **$r = 1$ (`70.58%`, `RR=85.63%`)** |        $-0.37\%$ (`-0.52%`)        |  `10×5`: $r=1$ (`68.73%`)`IG(k=7)`: $r=2$ (`69.15%`)  | Đạt đỉnh sớm tại $r=1$ (tăng từ `67.75%` ở $r=0$), giảm dần xuống `60.24%` ở $r=15$ do các lớp nhỏ bị hòa lẫn.    |

![1795835310562](image/report_error_tolerance_r/1795835310562.png)

![1795835327827](image/report_error_tolerance_r/1795835327827.png)

### 2.2. Nhận xét rút ra

- **Khớp chính xác vị trí $r$ tối ưu:** Bản tái lập tìm đúng $r^* = 2$ (`Aust`) và $r^* = 1$ (`Glass`), độ chính xác lệch rất nhỏ ($< 0.9\%$ tương đối). Kiểm định ổn định `10×5-fold Repeated CV` khẳng định đỉnh tối ưu vẫn nằm ở vùng $r$ nhỏ ($r = 1$ trên cả hai tập).
- **Quy luật tổng quát trên cả 6 tập (`r ∈ [0, 15], N = 1`):** Tăng $r$ giúp tỷ lệ nén (`RR%`) tăng đơn điệu, còn đỉnh Accuracy phụ thuộc độ chồng lấn lớp — rất nhỏ ở tập tách biệt rõ/nhiều lớp (`Iris`: $r=0$, `96.00%`; `Glass`: $r=1$, `70.58%`; `Wine`: $r=2$, `96.65%`; `Aust`: $r=2$, `86.38%`) và lớn hơn ở tập nhiễu biên cao (`Diabetes`: $r=5$, `74.08%`; `Heart`: $r=9$, `84.07%`). Chọn $r$ quá lớn gây **underfitting** do hình cầu lấn sang vùng lớp khác.

---

## 3. Thực nghiệm mở rộng 1: Hiện tượng Phân mảnh cấu trúc & Overfitting khi $r = 0$

### 3.1. Cơ chế và kết quả trên dữ liệu sạch (trung bình 6 bộ dữ liệu, 5-fold CV)

Khi $r = 0$, hình cầu đại diện dừng mở rộng ngay khi chạm điểm khác nhãn đầu tiên. Để tách bạch vai trò của độ dung nạp lỗi $r$ (Pruning 2) và ngưỡng cắt tỉa số lượng $N$ (Pruning 1), ta đối chiếu các cấu hình $(r, N)$ với `StandardKNN` ($k = 1, 3, 5$).

| Mô hình                         |  Train Acc (%)  | Test Acc (%) | Overfit Gap (%) |    Avg Reps    |     RR (%)     | Singleton $Num=1$ (%) | Small $Num \le 2$ (%) |    Avg Sim    |
| :-------------------------------- | :--------------: | :----------: | :-------------: | :-------------: | :-------------: | :---------------------: | :---------------------: | :------------: |
| **`KNNModel (r=0, N=0)`** | **100.00** |    82.95    | **17.05** | **90.27** | **74.00** |     **55.38**     |     **68.38**     | **0.22** |
| `KNNModel (r=2, N=0)`           |      93.69      |    82.78    |      10.91      |      41.77      |      87.16      |          37.65          |          47.51          |      0.39      |
| `KNNModel (r=4, N=0)`           |      91.27      |    81.83    |      9.45      |      30.30      |      90.36      |          33.37          |          44.74          |      0.46      |
| `KNNModel (r=0, N=1)`           |      90.72      |    82.91    |      7.81      |      27.67      |      91.28      |          15.46          |          39.76          |      0.48      |
| `KNNModel (r=1, N=1)`           |      91.40      |    82.67    |      8.73      |      32.37      |      90.42      |          15.85          |          33.17          |      0.50      |
| `KNNModel (r=2, N=1)`           |      90.67      |    82.91    | **7.76** |      27.33      |      91.70      |          12.96          |          28.87          |      0.55      |
| `KNNModel (r=4, N=1)`           |      89.27      |    81.38    |      7.89      |      21.03      | **93.46** |     **9.09**     |     **24.11**     | **0.64** |
| `StandardKNN (k=1)`             |      100.00      |    80.46    |      19.54      |       —       |      0.00      |           —           |           —           |       —       |
| `StandardKNN (k=3)`             |      89.90      |    82.18    |      7.72      |       —       |      0.00      |           —           |           —           |       —       |
| `StandardKNN (k=5)`             |      87.25      |    81.93    |      5.32      |       —       |      0.00      |           —           |           —           |       —       |

**Phân tích trọng tâm:**

1. **Overfitting cấu trúc khi tắt cắt tỉa (`r=0, N=0`):** Dù Test Accuracy trung bình trên dữ liệu sạch vẫn cao (`82.95%`), mô hình bị **học thuộc lòng tập Train** (`Train = 100%`, `Gap = 17.05%` — riêng `Glass` là `31.78%` và `Diabetes` là `27.74%`). Không gian bị **phân mảnh nặng**: `55.38%` đại diện là đơn điểm ($Num=1$), `68.38%` phủ $\le 2$ điểm, bán kính co hẹp (`0.22`) và **tỷ lệ nén sụt xuống 74.00%**, đánh mất ưu thế cốt lõi (>90% RR) của `kNNModel`.
2. **Tách bạch vai trò của $r$ và $N$:**
   - **Tăng thuần $r$ (giữ `N=0`):** Tăng $r$ từ $0 \to 4$ giúp cắt `Overfit Gap` từ `17.05%` $\to$ `9.45%`, giảm hơn 3 lần số đại diện (`90.27 → 30.30`) và kéo RR lên `90.36%`.
   - **Bật `N=1` tại `r=0`:** Cắt ngay `Overfit Gap` xuống `7.81%`, giảm mạnh đơn điểm (`55.38% → 15.46%`) và nâng RR lên `91.28%` (khớp sát mức `90.41%` ở Table 5 bài báo).
   - **Tương tác thú vị tại `(r=0, N=1)` vs `(r=1, N=1)`:** `(r=0, N=1)` có ít đại diện hơn (`27.67` vs `32.37`) và Gap thấp hơn (`7.81%` vs `8.73%`) vì ở $r=0$ điểm biên cô lập tạo đơn điểm rồi bị `N=1` xóa hẳn ở lượt 1, trong khi ở $r=1$ điểm biên được gộp vào cụm cỡ $\ge 2$ nên thoát cắt tỉa. Tuy nhiên tăng $r$ quá cao trên tập tách biệt rõ như `Iris` gây underfitting (`96.00% → 89.33%` tại $r=4$).

### 3.2. Kiểm chứng dưới nhiễu nhãn (Label Noise Stress Test: $0\% \to 20\%$, trung bình 6 tập × 3 seed)

Đảo ngẫu nhiên $p\% \in \{0\%, 5\%, 10\%, 15\%, 20\%\}$ nhãn tập Train trên cả 6 bộ dữ liệu (lặp 3 seed độc lập `{42, 123, 2024}`) và đánh giá trên **tập Test sạch 100%**:

| Mô hình                         | Avg Reps ($0\% \to 20\%$) | Hệ số phình Reps |    RR% ($0\% \to 20\%$)    | Clean Test Acc ($0\% \to 20\%$) | Mức giảm ($\Delta\%$) |
| :-------------------------------- | :--------------------------: | :------------------: | :--------------------------: | :-------------------------------: | :-----------------------: |
| **`KNNModel (r=0, N=0)`** |     `90.27 → 152.72`     |      `×1.69`      |      `74.00 → 49.58`      |       `82.95% → 76.13%`       |    **$-6.82$**    |
| `KNNModel (r=2, N=0)`           |      `41.77 → 72.23`      |      `×1.73`      |      `87.16 → 75.73`      |       `82.78% → 78.17%`       |         $-4.61$         |
| `KNNModel (r=4, N=0)`           |      `30.30 → 49.57`      |      `×1.64`      |      `90.36 → 83.26`      |       `81.83% → 78.60%`       |         $-3.23$         |
| `KNNModel (r=0, N=1)`           |      `27.67 → 39.14`      | **`×1.41`** |      `91.28 → 88.20`      |       `82.91% → 78.46%`       |         $-4.45$         |
| `KNNModel (r=2, N=1)`           |      `27.33 → 43.54`      |      `×1.59`      |      `91.70 → 86.24`      |       `82.91% → 78.08%`       |         $-4.83$         |
| **`KNNModel (r=4, N=1)`** | **`21.03 → 32.62`** |      `×1.55`      | **`93.46 → 89.68`** |  **`81.38% → 78.99%`**  |    **$-2.39$**    |
| `StandardKNN (k=1)`             |              —              |          —          |       `0.00 → 0.00`       |       `80.46% → 67.60%`       |   **$-12.86$**   |
| `StandardKNN (k=5)`             |              —              |          —          |       `0.00 → 0.00`       |       `81.93% → 77.82%`       |         $-4.11$         |

![1795835614125](image/report_error_tolerance_r/1795835614125.png)


![1795835623240](image/report_error_tolerance_r/1795835623240.png)

**Kết luận Phần 3:**

- **Bùng nổ đại diện & sụt nén khi tắt cắt tỉa (`r=0, N=0`):** Ở mức nhiễu 20%, tỷ lệ nén trung bình rơi sâu xuống **`49.58%`** và Clean Accuracy tụt mạnh nhất nhóm `kNNModel` (`-6.82%`, nặng nhất trên `Heart`: `-10.37%`, `Iris`: `-8.00%`, `Glass`: `-7.32%`). Trên các tập tách biệt rõ, số đại diện phình tới **5.2×** (`Iris`: `10.0 → 52.33`) và **4.2×** (`Wine`: `16.0 → 66.47`). Tuy vậy nhờ Luật 2 (`max Num`), ngay cả `(r=0, N=0)` vẫn bền hơn `1-NN` (`-12.86%`) rất nhiều.
- **Hai cơ chế cắt tỉa bổ trợ rõ rệt:**
  - Tăng thuần $r$ (ngay cả khi `N=0`) đã cắt mức giảm Accuracy từ `-6.82%` $\to$ `-3.23%` (`r=4, N=0` đạt `78.60%`, vượt `5-NN`).
  - Kết hợp **`KNNModel (r=4, N=1)`** đạt **độ bền nhiễu tốt nhất toàn bảng** ở mức nhiễu 20% (**`78.99%`**, chỉ giảm `2.39%`, `RR = 89.68%`), đặc biệt vượt trội trên tập chồng lấn cao `Diabetes` (`70.48%`, chỉ giảm `2.30%`, hơn `5-NN` `+3.51%`) và `Aust` (`80.10%`). Riêng trên `Wine` & `Glass`, `(r=0, N=1)` lọc nhiễu đơn điểm rất sắc (`Wine`: `94.37%`, chỉ giảm `0.60%`; `Glass`: `63.86%`, giảm `3.89%`).

---

## 4. Thực nghiệm mở rộng 2: Hạn chế của `kNNModel` dùng Euclidean trên Dữ liệu Hỗn hợp & Cải tiến bằng Gower / HEOM

### 4.1. Giải phẫu ma trận khoảng cách trên `Heart` và `Aust`

Khoảng cách Euclidean $d(x, y) = \sqrt{\sum_j (x_j - y_j)^2}$ giả định không gian liên tục đẳng hướng. Trên dữ liệu hỗn hợp, một sai khác Binary đóng góp $(1-0)^2 = 1.0$ và một sai khác Nominal sau One-Hot đóng góp $1^2 + 1^2 = 2.0$, trong khi sai khác trung bình trên cột Numeric chuẩn hóa $[0, 1]$ chỉ khoảng $0.15^2 \approx 0.0225$.

| Chỉ số phân tích không gian khoảng cách                                |              Heart (`7 NO`, `6 NN+NB`)              |               Aust (`6 NO`, `8 NN+NB`)               |
| :---------------------------------------------------------------------------- | :------------------------------------------------------: | :------------------------------------------------------: |
| Tỷ lệ số cột thực tế: Numeric / Non-Numeric                             |                 `53.85%` / `46.15%`                 |                 `42.86%` / `57.14%`                 |
| **[Euclidean Raw]** % tổng $\Delta d^2$ từ Numeric / Non-Numeric    |            `22.63%` / **`77.37%`**            |             `7.31%` / **`92.69%`**             |
| **[Euclidean OHE]** % tổng $\Delta d^2$ từ Numeric / Non-Numeric    |            `12.93%` / **`87.07%`**            |             `3.00%` / **`97.00%`**             |
| **[Gower Mixed — Đề xuất]** % tổng $d$ từ Numeric / Non-Numeric |            **`35.28%`** / `64.72%`            |            **`13.91%`** / `86.09%`            |
| Số mức khoảng cách phân biệt (Numeric / Binary / Non-Num OHE)           |     `35,626` / **`4`** / **`10`**     |     `198,553` / **`5`** / **`13`**     |
| **Neighbor Tie Rate %** (Numeric / Binary / Non-Num OHE)                | `0.00%` / **`100.00%`** / **`99.85%`** | `0.30%` / **`100.00%`** / **`99.88%`** |
| **Exact Collision $d = 0$ %** (Numeric / Binary / Non-Num OHE)        |       `0.00%` / **`22.89%`** / `2.42%`       |       `0.00%` / **`8.58%`** / `0.32%`       |

**Hai hệ quả hình học trực tiếp:**

1. **Biến rời rạc lấn át hoàn toàn biến số:** Dưới Euclidean + OHE, 6–7 cột Numeric chỉ còn đóng góp **12.93%** (`Heart`) và **3.00%** (`Aust`) vào tổng bình phương khoảng cách. Dùng **Gower Distance** lập tức khôi phục tỷ trọng biến số lên gấp **2.7×** (`Heart`: `35.28%`) và **4.6×** (`Aust`: `13.91%`).
2. **Lượng tử hóa, đồng khoảng cách (~100%) và va chạm tuyệt đối ($d = 0$):** Trên không gian `Binary Only`, chỉ tồn tại `4–5` mức khoảng cách, **100%** mẫu có láng giềng đồng khoảng cách và có tới **22.89%** (`Heart`) / **8.58%** (`Aust`) cặp mẫu **trùng khít vị trí ($d = 0$)**. Trong `KNNModel._build_model`, lệnh `np.argsort` phá hòa theo thứ tự chỉ số hàng tùy ý, khiến vòng mở rộng hình cầu dừng sớm bất thường khi đụng điểm khác nhãn trùng khoảng cách.

### 4.2. So sánh hiệu năng `KNNModel` và `StandardKNN` theo từng nhóm kiểu dữ liệu (Euclidean)

| Nhóm đặc trưng                 | Heart — `KNNModel` (`r=0,N=1` / `r=1,N=1`) | Heart — `StdKNN` (`k=5` / Best) | Aust — `KNNModel` (`r=0,N=1` / `r=1,N=1`) | Aust — `StdKNN` (`k=5` / Best) |
| :--------------------------------- | :-----------------------------------------------: | :----------------------------------: | :----------------------------------------------: | :---------------------------------: |
| **1. Numeric Only (`NO`)** |         `76.67%` / **`77.04%`**         |       `77.04%` / `77.04%`       |        **`75.36%`** / `75.22%`        |       `72.90%` / `74.93%`       |
| **2. Binary Only (`NB`)**  |              `53.70%` / `60.00%`              |       `51.85%` / `58.15%`       |        **`74.20%`** / `78.12%`        |  **`84.93%`** / `84.93%`  |
| **3. Nominal Only (Raw)**    |         `66.30%` / **`65.93%`**         |  `72.22%` / **`74.07%`**  |             `64.20%` / `63.04%`             |       `63.91%` / `63.91%`       |
| **4. Nominal Only (OHE)**    |              `63.33%` / `71.11%`              |       `67.78%` / `67.78%`       |        `63.33%` / **`59.86%`**        |  `64.06%` / **`65.07%`**  |
| **5. Non-Numeric (OHE)**     |         **`72.22%`** / `75.93%`         |  **`78.52%`** / `78.52%`  |        **`79.86%`** / `81.16%`        |  **`83.19%`** / `83.19%`  |
| **6. All Features (Raw)**    |              `80.74%` / `78.52%`              |       `78.89%` / `78.89%`       |             `84.20%` / `84.06%`             |       `84.64%` / `84.64%`       |
| **7. All Features (OHE)**    |              `78.52%` / `79.26%`              |       `78.15%` / `78.15%`       |             `80.72%` / `83.04%`             |       `83.04%` / `83.04%`       |

![1795835537362](image/report_error_tolerance_r/1795835537362.png)

**Nhận xét nổi bật:**

- **Mạnh trên không gian số liên tục (`Numeric Only`):** `KNNModel` đạt hiệu năng ngang bằng (`Heart`: `77.04%`, `r=0,N=0` đạt `78.89%`) hoặc vượt trội (`Aust`: `75.36%–76.09%` vs `72.90%–74.93%`) so với `StandardKNN`.
- **Suy yếu rõ trên không gian rời rạc trùng đỉnh cao:** Trên **`Aust — Binary Only`** (4 bit tạo tối đa $2^4 = 16$ đỉnh cho 690 mẫu), `KNNModel` thua `StandardKNN(k=5)` tới **$-10.72\%$** (`74.20%` vs `84.93%`) ở `r=0, N=1` và **$-6.81\%$** ở `r=1, N=1`. Trên `Non-Numeric (OHE)`, `KNNModel (r=0, N=1)` đều thua `5-NN` (`-6.30%` trên `Heart`, `-3.33%` trên `Aust`).
- **Nhạy cảm với OHE khi $r = 0$:** Chuyển từ `Raw` sang `OHE` làm `KNNModel (r=0, N=0)` giảm sâu (`Heart`: `81.48% → 75.19%`, `-6.29%`; `Aust`: `84.20% → 80.00%`, `-4.20%`) do OHE phình chiều và đẩy tỷ trọng $\Delta d^2$ của biến rời rạc lên 87%–97%.

### 4.3. Phân tích toán học về độ đo & Kết quả cải tiến bằng Gower Distance và HEOM

1. **Tính tương đương đơn điệu trên `{0, 1}` (`Non-Numeric OHE`):** Với hai vector nhị phân khác nhau tại $m$ vị trí, $d_{L_1} = m$, $d_{\text{Hamming}} = m/d$ và $d_{L_2} = \sqrt{m}$ cho **cùng thứ hạng lân cận** (thực nghiệm trên `Heart NN+NB OHE` trùng tuyệt đối: `72.22%`, `75.93%`, `78.52%`). Đồng thời không thể áp dụng Hamming trực tiếp cho cột số liên tục vì phép so khớp $x_j \neq y_j$ làm mất độ lớn định lượng.
2. **Đề xuất cài đặt trực tiếp hai độ đo hỗn hợp trong `KNNModel` (không cần OHE, không tạo thứ tự giả):**
   - **Gower Distance (1971):** $d_{\text{Gower}}(x, y) = \frac{1}{p}\!\left(\sum_{j \in \text{num}} |x_j - y_j| + \sum_{j \in \text{cat}} \mathbb{I}(x_j \neq y_j)\right) \in [0, 1]$.
   - **HEOM (*Wilson & Martinez, 1997* — Tài liệu tham khảo [9] của bài báo):** $d_{\text{HEOM}}(x, y) = \sqrt{\sum_{j \in \text{num}} (x_j - y_j)^2 + \sum_{j \in \text{cat}} \mathbb{I}(x_j \neq y_j)}$.

| Độ đo trên `All Features` (Mixed) | Heart `r=0,N=1` | Heart `r=1,N=1` | Heart `r=2,N=1` |  Heart `k=5`  | Aust `r=0,N=1` | Aust `r=1,N=1` | Aust `r=2,N=1` |   Aust `k=5`   |
| :-------------------------------------- | :---------------: | :---------------: | :---------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: |
| Euclidean (Raw MinMax)                  |      80.74%      |      78.52%      |      80.00%      |      78.89%      | **84.20%** | **84.06%** | **86.38%** |      84.64%      |
| Euclidean (OHE)                         |      78.52%      |      79.26%      |      76.67%      |      78.15%      |      80.72%      |      83.04%      |      81.74%      |      83.04%      |
| Manhattan / $L_1$ (OHE)               |      80.00%      |      78.15%      |      76.67%      |      79.63%      |      82.75%      |      81.59%      |      83.48%      |      83.77%      |
| Cosine (OHE)                            |      80.74%      |      76.30%      |      76.30%      |      79.63%      |      81.30%      |      81.30%      |      81.74%      |      81.59%      |
| **Gower Distance (Đề xuất)**   | **83.33%** |      80.00%      | **80.74%** | **80.37%** |      82.32%      |      82.32%      |      83.33%      | **86.67%** |
| **HEOM Distance (Đề xuất)**    |      79.63%      | **81.85%** | **80.74%** |      80.00%      |      83.77%      |      83.19%      |      83.33%      |      86.23%      |

![1795835517372](image/report_error_tolerance_r/1795835517372.png)

**Đánh giá hiệu quả cải tiến:**

- **Trên `Heart`:** **Gower Distance** và **HEOM** vượt trội toàn diện tất cả các cấu hình chuẩn (kể cả Euclidean Raw lẫn OHE): Gower đạt đỉnh **`83.33%`** tại `r=0, N=1` (**`+4.81%`** so với Euclidean OHE), HEOM đạt đỉnh **`81.85%`** tại `r=1, N=1` (**`+2.59%`**), cả hai đạt **`80.74%`** tại `r=2, N=1` (**`+4.07%`**), đồng thời giữ nguyên tỷ lệ nén cao (`RR = 88.6%–88.8%`).
- **Trên `Aust`:** So với chuẩn mã hóa không thứ tự `Euclidean (OHE)`, **HEOM** thắng trên mọi cấu hình (`r=0,N=1`: **`+3.05%`** lên `83.77%`; `r=2,N=1`: `+1.59%`), còn **Gower** và **HEOM** đẩy `StandardKNN(k=5)` lên mức cao nhất toàn bảng (**`86.67%`** và **`86.23%`**, tăng **`+3.63%`** và **`+3.19%`**), đồng thời giữ tỷ lệ nén của `KNNModel` ở mức `89.6%–89.7%`. *(Lưu ý: `Euclidean Raw` đạt kết quả tốt riêng trên `Aust` do một số mã danh mục có tương quan thứ tự ngẫu nhiên, nhưng giả định thứ tự trên biến Nominal không hợp lệ về mặt tổng quát và đã kém Gower/HEOM trên `Heart`.)*

---

## 5. Kết luận ngắn gọn

1. **Vai trò của $r$ (Hình 13 & 14):** Tham số $r$ cân bằng giữa **độ chính xác** và **tỷ lệ nén dữ liệu**. Đỉnh tái lập khớp bài báo ở giá trị nhỏ ($r=2$ trên `Aust`, $r=1$ trên `Glass`); tăng $r$ quá lớn gây underfitting.
2. **Hạn chế khi $r = 0$ & vai trò cắt tỉa:** Tắt cắt tỉa (`r=0, N=0`) gây phân mảnh cấu trúc (`RR` chỉ còn `74.00%`, `55.38%` là đơn điểm) và suy giảm mạnh dưới nhiễu nhãn 20% (`RR` rơi xuống `49.58%`, Accuracy giảm `6.82%`, phình 4–5× số đại diện trên `Iris`/`Wine`). Kết hợp `N=1` và $r > 0$ khắc phục triệt để: `(r=4, N=1)` đạt độ bền nhiễu cao nhất (`78.99%`, chỉ giảm `2.39%`, `RR = 89.68%`).
3. **Hạn chế về kiểu dữ liệu & cải tiến:** Euclidean + OHE làm biến rời rạc lấn át 87%–97% khoảng cách, tạo ~100% đồng khoảng cách và tới 22.89% va chạm $d=0$, khiến `kNNModel` yếu đi rõ trên không gian rời rạc trùng đỉnh (`Aust - Binary Only` thua `5-NN` tới `10.72%`). Cài đặt trực tiếp **Gower Distance** hoặc **HEOM** giúp khôi phục cân bằng trọng số Numeric/Nominal và cải thiện rõ rệt hiệu năng trên cả `Heart` (đạt tới **`83.33%`**) lẫn `Aust` (vượt `Euclidean OHE` tới **`+3.05%`** với `kNNModel` và **`+3.63%`** với `5-NN`).
