# Báo cáo ngắn - Day 21 CI/CD cho AI Systems

## 1. Bộ siêu tham số đã chọn

Mô hình sử dụng `RandomForestClassifier` với bộ siêu tham số cuối cùng:

- `n_estimators: 800`
- `max_depth: null`
- `min_samples_split: 2`

Bộ tham số này được chọn sau khi chạy nhiều thí nghiệm cục bộ và so sánh trên MLflow UI. Kết quả hiện tại trên tập đánh giá là:

- `accuracy`: 0.758
- `f1_score`: 0.7570

Lý do chọn: đây là cấu hình đạt độ chính xác trên ngưỡng yêu cầu `0.70`, đồng thời có `f1_score` tương ứng ổn định trên tập đánh giá. Việc đặt `max_depth: null` cho phép các cây học được quan hệ phức tạp hơn trong dữ liệu, còn `n_estimators: 800` giúp mô hình giảm dao động so với số lượng cây nhỏ hơn.

## 2. Pipeline CI/CD

Pipeline GitHub Actions gồm bốn job bắt buộc:

1. `Unit Test`: chạy unit test cho logic huấn luyện.
2. `Train`: kéo dữ liệu bằng DVC, huấn luyện mô hình, ghi `metrics.json`, upload model lên Cloud Storage.
3. `Eval`: kiểm tra `accuracy >= 0.70`.
4. `Deploy`: restart service FastAPI trên VM và kiểm tra endpoint `/health`.

Pipeline được cấu hình để chạy khi có thay đổi ở code, tham số, workflow, test hoặc file con trỏ DVC.

## 3. Khó khăn và cách giải quyết

Khó khăn chính là phối hợp giữa Git, DVC, Cloud Storage và GitHub Actions. File dữ liệu CSV thật không được commit vào Git mà được quản lý bằng DVC, vì vậy cần chạy `dvc push` trước khi `git push` commit chứa file `.dvc`. Nếu làm ngược lại, GitHub Actions có thể bắt đầu trước khi dữ liệu mới có trên cloud và job `dvc pull` sẽ lỗi.

Một điểm cần chú ý khác là deploy lên VM chỉ hoạt động khi GitHub Secrets đã được cấu hình đầy đủ gồm credentials cloud, bucket, host VM, user VM và SSH private key.
