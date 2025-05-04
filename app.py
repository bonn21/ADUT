import os
import time
import cv2
import torch
import torch.nn as nn
import timm
from torchvision import transforms
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from threading import Thread, Lock
from collections import deque

app = Flask(__name__)

# Định nghĩa transform cho frame webcam
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Định nghĩa label map
label_map = {
    0: 'calling', 1: 'clapping', 2: 'cycling', 3: 'dancing', 4: 'drinking',
    5: 'eating', 6: 'fighting', 7: 'hugging', 8: 'laughing', 9: 'listening_to_music',
    10: 'running', 11: 'sitting', 12: 'sleeping', 13: 'texting', 14: 'using_laptop'
}

# Khởi tạo model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Thay đổi kiến trúc mô hình từ swin sang vit
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=15)

# Load model
checkpoint = torch.load('../file_final/model_ver2.pth', map_location=device)
state_dict = checkpoint.get('model_state_dict', checkpoint)
clean_state_dict = {}
for k, v in state_dict.items():
    new_key = k
    if k.startswith('vit.'):
        new_key = k[len('vit.'):]
    elif k.startswith('module.vit.'):
        new_key = k[len('module.vit.'):]
    clean_state_dict[new_key] = v

# Load với strict=False để bỏ qua các key không khớp
model.load_state_dict(clean_state_dict, strict=False)
model = model.to(device)
model.eval()

# Biến toàn cục
camera = cv2.VideoCapture(0)  # Sử dụng camera mặc định (1)
latest_frame = None
latest_prediction = "None"
latest_confidence = 0.0
lock = Lock()
running = True
predict_interval = 0.2
confidence_threshold = 0.7
label_history = deque(maxlen=5)  # Giữ 5 nhãn gần nhất để ổn định kết quả

def model_inference():
    global latest_prediction, latest_confidence, running, latest_frame
    last_predict_time = 0
    
    while running:
        current_time = time.time()
        if current_time - last_predict_time < predict_interval:
            time.sleep(0.001)
            continue

        with lock:
            if latest_frame is None:
                continue
            frame_rgb = latest_frame.copy()

        # Chuẩn bị frame cho mô hình
        input_tensor = transform(frame_rgb).unsqueeze(0).to(device)

        # Dự đoán với mô hình
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        # Kiểm tra ngưỡng và cập nhật dự đoán
        with lock:
            if confidence >= confidence_threshold:
                prediction = label_map[predicted_class]
                label_history.append(prediction)
                # Lấy nhãn xuất hiện nhiều nhất trong lịch sử
                latest_prediction = max(set(label_history), key=label_history.count)
                latest_confidence = confidence
            else:
                latest_prediction = "Not determined"
                latest_confidence = confidence

        last_predict_time = current_time

def gen_frames():
    global latest_frame, latest_prediction, latest_confidence
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Chuyển đổi frame sang RGB cho model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Cập nhật frame mới nhất cho model
            with lock:
                latest_frame = frame_rgb
                current_prediction = latest_prediction
                current_confidence = latest_confidence

            # Hiển thị kết quả trên frame
            label_text = f"{current_prediction}: {current_confidence:.2f}"
            cv2.putText(frame, label_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode frame để stream
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    global latest_frame, latest_prediction, latest_confidence
    with lock:
        prediction = latest_prediction
        confidence = float(latest_confidence)  # Convert to float for JSON serialization
    
    return jsonify({
        'prediction': prediction,
        'confidence': confidence
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/shutdown', methods=['POST'])
def shutdown():
    global running
    running = False
    # Giải phóng camera
    camera.release()
    # Tắt server Flask
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'

if __name__ == '__main__':
    # Khởi động luồng dự đoán
    inference_thread = Thread(target=model_inference)
    inference_thread.daemon = True  # Đảm bảo thread sẽ dừng khi chương trình chính dừng
    inference_thread.start()
    
    # Khởi động Flask app
    app.run(debug=True, use_reloader=False)  # Tắt reloader để tránh chạy nhiều thread