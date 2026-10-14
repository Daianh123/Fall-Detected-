# Canh - Phát hiện té ngã

Ứng dụng web Flask nhận video từ camera RTSP/RTSPS hoặc tệp video, dùng YOLOv8 Pose để trích xuất keypoint và LSTM để phân loại té ngã.

## Yêu cầu

- Windows và Python 3.10 được khuyến nghị.
- Hai tệp trong thư mục gốc dự án: `fall_lstm_model.pth` và `yolov8n-pose.pt`.
- `output_result.mp4` là video mẫu trên trang chủ; nếu thiếu, trang vẫn chạy nhưng hiện khung trống.
- Camera RTSP phải truy cập được từ máy chạy Flask.

## Cài đặt

Mở PowerShell tại thư mục dự án và chạy:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu đã có môi trường `.venv`, bỏ qua lệnh tạo mới. Dependencies được khai báo trong `requirements.txt`.

## Chạy web

```powershell
.\.venv\Scripts\python.exe .\web_app.py
```

Mở `http://127.0.0.1:5000/` trong trình duyệt. Nhấn `Ctrl+C` trong terminal để dừng Flask. Đây là server phát triển, không nên đưa trực tiếp lên Internet.

## Sử dụng

1. Từ trang giới thiệu, chọn **Mở bàn giám sát**.
2. Chọn tab **Camera RTSP** và nhập địa chỉ, ví dụ `rtsp://192.168.1.20:554/stream`; hoặc chọn tab **Tệp video** và tải lên tệp dưới 500 MB.
3. Nhấn **Bắt đầu phát hiện**. Dùng nút dừng để kết thúc phiên.

Web dùng YOLOv8 Pose lấy 17 điểm khớp (34 tọa độ) mỗi frame. Khi có đủ 30 frame, LSTM đưa ra một trong hai lớp: té ngã hoặc bình thường. Trang hiển thị khung hình đã vẽ keypoint, nhãn và độ tin cậy. Tệp tải lên được xử lý trên máy chạy web và tự xóa sau khi xử lý kết thúc.

## Thay video mẫu trang chủ

Đặt video trong thư mục gốc, sau đó sửa `HOME_DEMO_PATH` trong `web_app.py`, ví dụ:

```python
HOME_DEMO_PATH = BASE_DIR / "video_mau.mp4"
```

Lưu file và khởi động lại Flask. OpenCV đọc video mẫu và phát thành luồng JPEG để trình duyệt hiển thị được cả video có codec gốc không được hỗ trợ.

## Tệp chính

| Tệp                      | Vai trò                                |
| ------------------------ | -------------------------------------- |
| `web_app.py`             | Flask routes, nạp model và xử lý video |
| `templates/home.html`    | Trang giới thiệu                       |
| `templates/monitor.html` | Trang bàn giám sát                     |
| `static/style.css`       | Giao diện và bố cục responsive         |
| `static/monitor.js`      | Tab nguồn, gọi API và cập nhật kết quả |
| `fall_lstm_model.pth`    | Trọng số LSTM                          |
| `yolov8n-pose.pt`        | Trọng số YOLOv8 Pose                   |

## Thiết bị chạy

Ứng dụng tự dùng CUDA nếu PyTorch nhận diện được GPU; nếu không, model chạy trên CPU và có thể chậm hơn. Trang hiện chỉ lắng nghe trên `127.0.0.1`, phù hợp để chạy và thử nghiệm trên máy local.
