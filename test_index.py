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

model.load_state_dict(clean_state_dict, strict=False)
model = model.to(device)
model.eval()

# Load YOLOv5 để phát hiện đối tượng
import yolov5
yolo_model = yolov5.load('yolov5s.pt')
yolo_model.conf = 0.5

# Biến toàn cục
camera = cv2.VideoCapture(0)
latest_frame = None
latest_prediction = "None"
latest_confidence = 0.0
latest_box = None
action_start_time = None
action_end_time = None
lock = Lock()
running = True
predict_interval = 0.2
confidence_threshold = 0.7
label_history = deque(maxlen=5)
action_history = deque(maxlen=10)

def model_inference():
    global latest_prediction, latest_confidence, latest_box, running, latest_frame, action_start_time, action_end_time
    last_predict_time = 0
    current_action = None

    while running:
        current_time = time.time()
        if current_time - last_predict_time < predict_interval:
            time.sleep(0.001)
            continue

        with lock:
            if latest_frame is None:
                continue
            frame_rgb = latest_frame.copy()

        results = yolo_model(frame_rgb)
        detections = results.xyxy[0].cpu().numpy()
        person_box = None
        for det in detections:
            if int(det[5]) == 0:
                x_min, y_min, x_max, y_max = map(int, det[:4])
                person_box = (x_min, y_min, x_max, y_max)
                break

        if person_box is None:
            input_tensor = transform(frame_rgb).unsqueeze(0).to(device)
        else:
            x_min, y_min, x_max, y_max = person_box
            person_img = frame_rgb[y_min:y_max, x_min:x_max]
            if person_img.size == 0:
                continue
            input_tensor = transform(person_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        with lock:
            if confidence >= confidence_threshold:
                prediction = label_map[predicted_class]
                label_history.append(prediction)
                latest_prediction = max(set(label_history), key=label_history.count)
                latest_confidence = confidence
                latest_box = person_box
            else:
                latest_prediction = "Not determined"
                latest_confidence = confidence
                latest_box = None

            action_history.append(latest_prediction)
            if len(action_history) == action_history.maxlen:
                most_common_action = max(set(action_history), key=action_history.count)
                if most_common_action != "Not determined":
                    if current_action != most_common_action:
                        current_action = most_common_action
                        action_start_time = current_time
                        action_end_time = None
                    else:
                        action_end_time = current_time
                else:
                    current_action = None
                    action_start_time = None
                    action_end_time = None

        last_predict_time = current_time

def gen_frames():
    global latest_frame, latest_prediction, latest_confidence, latest_box, action_start_time, action_end_time
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            with lock:
                latest_frame = frame_rgb
                current_prediction = latest_prediction
                current_confidence = latest_confidence
                current_box = latest_box
                start_time = action_start_time
                end_time = action_end_time

            if current_box is not None:
                x_min, y_min, x_max, y_max = current_box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Chỉ hiển thị nhãn hành động và độ chính xác trên video
            label_text = f"{current_prediction}: {current_confidence:.2f}"
            cv2.putText(frame, label_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    global latest_prediction, latest_confidence, latest_box, action_start_time, action_end_time
    with lock:
        prediction = latest_prediction
        confidence = float(latest_confidence)
        box = latest_box if latest_box is not None else []
        start_time = float(action_start_time) if action_start_time is not None else None
        end_time = float(action_end_time) if action_end_time is not None else None
    
    return jsonify({
        'prediction': prediction,
        'confidence': confidence,
        'box': box,
        'start_time': start_time,
        'end_time': end_time
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/shutdown', methods=['POST'])
def shutdown():
    global running
    running = False
    camera.release()
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'

if __name__ == '__main__':
    inference_thread = Thread(target=model_inference)
    inference_thread.daemon = True
    inference_thread.start()
    app.run(debug=True, use_reloader=False)