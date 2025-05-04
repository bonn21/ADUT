# Dự án Nhận dạng Hành động Thời gian thực

Dự án này triển khai hệ thống nhận dạng hành động thời gian thực sử dụng webcam. Hệ thống sẽ quay video, xử lý các khung hình thông qua mô hình nhận dạng hành động đã được huấn luyện trước, và hiển thị các hành động đã phát hiện cùng với hộp giới hạn xung quanh các hành động được nhận dạng.

## Cấu trúc dự án

```
real-time-action-recognition/
├── src/
│   ├── app.py                     # Điểm khởi chạy chính cho ứng dụng Flask
│   ├── webcam.py                  # Xử lý quay video và nhận dạng hành động
│   ├── models/
│   │   └── action_recognition_model.py  # Định nghĩa mô hình nhận dạng hành động
│   ├── utils/
│   │   ├── preprocessing.py        # Các hàm tiền xử lý khung hình video
│   │   └── visualization.py        # Các hàm hiển thị hành động đã phát hiện
│   └── config/
│       └── settings.py            # Cài đặt cấu hình cho ứng dụng
├── requirements.txt                # Danh sách các phụ thuộc của dự án
├── README.md                       # Tài liệu dự án
└── .gitignore                      # Chỉ định các tệp cần bỏ qua trong Git
```

## Hướng dẫn cài đặt

1. **Sao chép kho lưu trữ:**
   ```
   git clone <repository-url>
   cd real-time-action-recognition
   ```

2. **Tạo môi trường ảo (không bắt buộc nhưng được khuyến nghị):**
   ```
   python -m venv venv
   source venv/bin/activate  # Trên Windows sử dụng `venv\Scripts\activate`
   ```

3. **Cài đặt các phụ thuộc cần thiết:**
   ```
   pip install -r requirements.txt
   ```

## Cách sử dụng

1. **Chạy ứng dụng Flask:**
   ```
   python src/app.py
   ```

2. **Mở trình duyệt web và điều hướng đến:**
   ```
   http://127.0.0.1:5000
   ```

3. **Để bắt đầu nhận dạng hành động thời gian thực, chạy script webcam:**
   ```
   python src/webcam.py
   ```

## Phụ thuộc

Dự án sử dụng các thư viện chính sau:
- Flask
- OpenCV
- PyTorch
- torchvision
- timm

## Yêu cầu hệ thống

- Python 3.7 trở lên
- GPU hỗ trợ CUDA (khuyến nghị cho hiệu suất tốt hơn)
- Webcam hoặc camera kết nối với máy tính

## Tính năng chính

- Nhận dạng hành động theo thời gian thực từ webcam
- Giao diện web đơn giản để hiển thị kết quả
- Hỗ trợ nhiều loại hành động khác nhau
- Hiển thị trực quan các hành động được phát hiện

