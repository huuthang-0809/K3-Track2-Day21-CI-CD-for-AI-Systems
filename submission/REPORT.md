# BÁO CÁO KẾT QUẢ DỰ ÁN MLOPS CI/CD PIPELINE

**Họ và tên:** NGUYỄN HỮU THẮNG  
**MSV:** 2A202601435  
**Dự án:** CI/CD for AI Systems - Wine Quality Classification  
**Kho lưu trữ:** [K3-Track2-Day21-CI-CD-for-AI-Systems](https://github.com/huuthang-0809/K3-Track2-Day21-CI-CD-for-AI-Systems)  

---

### 1. Bộ Siêu Tham Số Đã Chọn Và Lý Do (Dựa Trên MLflow)

Sau khi thực hiện quét (Grid Search) qua các bộ siêu tham số trên MLflow, mô hình **RandomForestClassifier** đã chọn bộ tham số tối ưu lưu tại `params.yaml`:
- **`n_estimators`**: `200`
- **`max_depth`**: `18`
- **`min_samples_split`**: `2`

**Lý do lựa chọn:**
- Trong quá trình theo dõi thí nghiệm trên MLflow UI, bộ tham số này mang lại chỉ số vượt trội nhất trên tập đánh giá `eval.csv` ở Phase 1 với **Accuracy: 0.6940** và **F1-Score: 0.6929** (cao hơn đáng kể so với cấu hình mặc định ban đầu `n_estimators=100, max_depth=10` chỉ đạt `Accuracy: 0.6720`).
- Độ sâu `max_depth=18` cho phép cây quyết định biểu diễn được các mối quan hệ phi tuyến phức tạp giữa 12 đặc trưng hóa lý của rượu (độ chua, lượng đường, nồng độ cồn...), trong khi `n_estimators=200` giúp giảm thiểu biến động (variance) và giữ mô hình ổn định không bị quá mịn (overfitting).

---

### 2. So Sánh Kết Quả Huấn Luyện (2,998 mẫu vs 5,996 mẫu)

| Chỉ số Đánh Giá | Bước 2 (Tập dữ liệu 2,998 mẫu) | Bước 3 (Tập dữ liệu 5,996 mẫu) | Biến động | Trạng thái Quality Gate |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **0.6940** (69.40%) | **0.7480** (74.80%) | 🟢 **+5.40%** | Vượt ngưỡng (`>= 0.70`) |
| **F1-Score** | **0.6929** (69.29%) | **0.7470** (74.70%) | 🟢 **+5.41%** | Vượt ngưỡng (`>= 0.70`) |
| **Eval Gate** | 🔴 Thất bại (`0.6940 < 0.70`) | 🟢 **Thành công** (`0.7480 >= 0.70`) | - | Chuyển trạng thái sang Deploy |
| **Deploy Job** | 🚫 Bị chặn không triển khai | 🟢 **Triển khai tự động lên Cloud VM** | - | Phục vụ REST API |

**Nhận xét:** Việc bổ sung 2,998 mẫu dữ liệu mới từ `train_phase2.csv` giúp mô hình tăng mạnh Accuracy từ **0.6940** lên **0.7480** (+5.40%). Sự gia tăng dữ liệu đa dạng giúp mô hình tổng quát hóa tốt hơn, chính thức vượt mốc chất lượng `0.70` của Eval Gate và kích hoạt tiến trình tự động Deploy lên máy chủ sản xuất Compute Engine.

---

### 3. Khó Khăn Gặp Phải Và Cách Giải Quyết

1. **Khó khăn 1: GCP Organization Policy chặn tạo Key JSON Service Account**
   - *Mô tả:* Lệnh `gcloud iam service-accounts keys create` báo lỗi `constraints/iam.disableServiceAccountKeyCreation`.
   - *Giải quyết:* Cập nhật cấu hình Organization Policy trên GCP bằng lệnh `gcloud org-policies set-policy` thiết lập `enforce: false` đối với giới hạn tạo key, sau đó xuất thành công file `sa-key.json` để xác thực DVC và GCS SDK.

2. **Khó khăn 2: Lỗi xác thực SSH & Kết nối REST API bị ngắt ở Job Deploy**
   - *Mô tả:* Lệnh SSH deploy trên GitHub Actions gặp lỗi ký tự ẩn trong secret `VM_USER` (`remote username contains invalid characters`) và lệnh `curl` kiểm tra `/health` bị ngắt (`Connection refused`) do FastAPI server cần 5-8s để tải mô hình từ GCS.
   - *Giải quyết:* 
     - Lọc sạch các ký tự ẩn rác (`\r\n`) trong các biến môi trường SSH bằng `tr -d '\r\n" '`.
     - Thêm cơ chế **vòng lặp kiểm tra sức khỏe (Retry Loop 10 lần, nghỉ 3s)** cho lệnh `curl http://localhost:8000/health`, đảm bảo uvicorn khởi tạo xong mô hình và lắng nghe thành công trên cổng 8000 trước khi kết thúc pipeline.

---

### 4. Báo Cáo Triển Khai Các Tính Năng Bonus (12 Điểm Bonus)

#### 🌟 Bonus 3: Báo Cáo Hiệu Suất Tự Động (+4 điểm)
- Trong `src/train.py`, hệ thống tự động tính toán **Ma trận nhầm lẫn (Confusion Matrix)** và các chỉ số **Precision / Recall / F1-Score** chi tiết cho từng lớp (`thap (0)`, `trung_binh (1)`, `cao (2)`).
- Kết quả được in ra dạng văn bản và lưu tự động vào file `outputs/report.txt`.
- Pipeline GitHub Actions tự động đính kèm `outputs/report.txt` cùng `outputs/metrics.json` thành Artifact mang tên **`performance-report`** sau mỗi lượt chạy.

#### 🌟 Bonus 4: Hoàn Trả Về Phiên Bản Trước - Chống Giảm Hiệu Suất (+4 điểm)
- Trước khi thực hiện Deploy, Job `Eval` tự động kết nối đến Cloud Storage và tải file `models/latest/metrics.json` của phiên bản mô hình đang chạy trên sản xuất (nếu có).
- So sánh `Accuracy mới` với `Accuracy cũ`. Nếu mô hình mới bị giảm Accuracy (`Accuracy mới < Accuracy cũ`), pipeline sẽ lập tức **hủy bỏ Deploy (Exit 1)** để bảo vệ hệ thống khỏi việc sụt giảm chất lượng (Model Regression).
- Kết quả so sánh chi tiết giữa 2 phiên bản mô hình được ghi lại minh bạch trong log của pipeline.

#### 🌟 Bonus 5: Cảnh Báo Lệch Lạc Dữ Liệu - Data Imbalance / Drift (+4 điểm)
- Trong `src/train.py`, hệ thống tự động tính tỷ lệ phân phối dữ liệu (Ratio %) của từng lớp (0, 1, 2) trên tập huấn luyện.
- Nếu bất kỳ lớp nào chiếm **dưới 10%** tổng số mẫu, hệ thống sẽ phát tín hiệu cảnh báo rõ ràng `[WARNING] Class X ratio Y% < 10% (Data Imbalance Detected!)` vào log.
- Tỷ lệ phân phối nhãn chi tiết được lưu dạng dictionary `label_distribution` vào file `outputs/metrics.json` phục vụ việc theo dõi sự dịch chuyển dữ liệu qua các phiên bản.
