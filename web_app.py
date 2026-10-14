import os
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "fall_lstm_model.pth"
POSE_PATH = BASE_DIR / "yolov8n-pose.pt"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
HOME_DEMO_PATH = BASE_DIR / "output_result.mp4"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


class FallLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_layers=2, num_classes=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, sequence):
        hidden = torch.zeros(self.num_layers, sequence.size(0), self.hidden_size, device=sequence.device)
        cell = torch.zeros(self.num_layers, sequence.size(0), self.hidden_size, device=sequence.device)
        output, _ = self.lstm(sequence, (hidden, cell))
        return self.fc(output[:, -1, :])


class DetectionService:
    def __init__(self):
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.pose_model = None
        self.fall_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state = self._empty_state()
        self.frame = None

    @staticmethod
    def _empty_state():
        return {
            "running": False,
            "phase": "idle",
            "label": "Chưa kết nối",
            "confidence": None,
            "frames": 0,
            "source": "",
            "error": None,
        }

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def latest_frame(self):
        with self.lock:
            return self.frame

    def start(self, source, source_label):
        with self.lock:
            if self.state["running"]:
                return False
            self.stop_event = threading.Event()
            self.frame = None
            self.state = {
                "running": True,
                "phase": "loading",
                "label": "Đang nạp mô hình",
                "confidence": None,
                "frames": 0,
                "source": source_label,
                "error": None,
            }
            thread = threading.Thread(
                target=self._run,
                args=(source, self.stop_event),
                daemon=True,
                name="fall-detection",
            )
            thread.start()
            return True

    def stop(self):
        self.stop_event.set()

    def _update(self, **values):
        with self.lock:
            self.state.update(values)

    def _load_models(self):
        if self.pose_model is None:
            if not POSE_PATH.is_file():
                raise FileNotFoundError(f"Không tìm thấy trọng số YOLO: {POSE_PATH.name}")
            self.pose_model = YOLO(str(POSE_PATH))
        if self.fall_model is None:
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(f"Không tìm thấy trọng số LSTM: {MODEL_PATH.name}")
            fall_model = FallLSTM().to(self.device)
            weights = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
            fall_model.load_state_dict(weights)
            fall_model.eval()
            self.fall_model = fall_model
        return self.pose_model, self.fall_model

    def _run(self, source, stop_event):
        capture = None
        temporary_upload = isinstance(source, Path) and UPLOAD_DIR in source.parents
        try:
            self._update(phase="loading", label="Đang nạp mô hình")
            pose_model, fall_model = self._load_models()
            if stop_event.is_set():
                return

            capture = cv2.VideoCapture(str(source))
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not capture.isOpened():
                raise RuntimeError("Không mở được nguồn video. Kiểm tra địa chỉ RTSP hoặc định dạng tệp.")

            sequence = deque(maxlen=30)
            self._update(phase="connecting", label="Đang đọc hình ảnh")
            while not stop_event.is_set():
                has_frame, frame = capture.read()
                if not has_frame:
                    break

                results = pose_model(frame, verbose=False)
                keypoints_result = results[0].keypoints
                if keypoints_result is not None and len(keypoints_result.xy) > 0:
                    keypoints = keypoints_result.xy[0].cpu().numpy().reshape(-1)
                    if keypoints.size != 34:
                        keypoints = np.zeros(34, dtype=np.float32)
                    annotated = results[0].plot()
                else:
                    keypoints = np.zeros(34, dtype=np.float32)
                    annotated = frame.copy()

                sequence.append(keypoints)
                label = "Đang gom dữ liệu"
                confidence = None
                phase = "warming"
                color = (64, 190, 224)

                if len(sequence) == 30:
                    tensor = torch.tensor(np.asarray(sequence), dtype=torch.float32).unsqueeze(0).to(self.device)
                    with torch.inference_mode():
                        probabilities = functional.softmax(fall_model(tensor), dim=1)
                        class_index = int(torch.argmax(probabilities, dim=1).item())
                        confidence = float(probabilities[0, class_index].item())
                    label = "Phát hiện té ngã" if class_index == 1 else "Bình thường"
                    phase = "alert" if class_index == 1 else "monitoring"
                    color = (56, 64, 220) if class_index == 1 else (72, 157, 68)

                cv2.rectangle(annotated, (16, 16), (370, 66), (20, 31, 31), -1)
                cv2.putText(annotated, label, (30, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                encoded, image = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if encoded:
                    with self.lock:
                        self.frame = image.tobytes()
                        self.state.update(
                            phase=phase,
                            label=label,
                            confidence=confidence,
                            frames=self.state["frames"] + 1,
                        )

            if not stop_event.is_set():
                self._update(phase="complete", running=False, label="Đã xử lý xong")
        except Exception as error:
            self._update(phase="error", running=False, label="Không thể chạy phát hiện", error=str(error))
        finally:
            if capture is not None:
                capture.release()
            if temporary_upload:
                try:
                    source.unlink(missing_ok=True)
                except OSError:
                    pass
            if stop_event.is_set():
                self._update(phase="stopped", running=False, label="Đã dừng")


detection = DetectionService()


@app.get("/")
def home():
    return render_template("home.html", demo_available=HOME_DEMO_PATH.is_file())


@app.get("/monitor")
def monitor():
    return render_template("monitor.html")


@app.get("/demo")
def demo_video():
    if not HOME_DEMO_PATH.is_file():
        return "", 404

    capture = cv2.VideoCapture(str(HOME_DEMO_PATH))
    if not capture.isOpened():
        capture.release()
        return "", 404
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_delay = 1 / fps if fps > 0 else 1 / 24

    def stream_demo():
        try:
            while True:
                has_frame, frame = capture.read()
                if not has_frame:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                encoded, image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if encoded:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + image.tobytes() + b"\r\n"
                    time.sleep(frame_delay)
        finally:
            capture.release()

    return Response(
        stream_demo(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/start")
def start_detection():
    video_file = request.files.get("video")
    if video_file and video_file.filename:
        suffix = Path(video_file.filename).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            return jsonify(error="Định dạng video chưa được hỗ trợ."), 400
        UPLOAD_DIR.mkdir(exist_ok=True)
        safe_name = secure_filename(video_file.filename) or "video"
        upload_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
        video_file.save(upload_path)
        source = upload_path
        source_label = safe_name
    else:
        source = request.form.get("source", "").strip()
        parsed = urlparse(source)
        if parsed.scheme not in {"rtsp", "rtsps"} or not parsed.hostname:
            return jsonify(error="Nhập địa chỉ RTSP/RTSPS hợp lệ, ví dụ rtsp://camera.local:554/stream."), 400
        source_label = "Camera RTSP"

    if not detection.start(source, source_label):
        if isinstance(source, Path):
            source.unlink(missing_ok=True)
        return jsonify(error="Đang có một phiên phát hiện chạy. Hãy dừng phiên đó trước."), 409
    return jsonify(ok=True), 202


@app.post("/api/stop")
def stop_detection():
    detection.stop()
    return jsonify(ok=True)


@app.get("/api/status")
def detection_status():
    return jsonify(detection.snapshot())


@app.get("/video_feed")
def video_feed():
    def stream_frames():
        while True:
            frame = detection.latest_frame()
            if frame is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                time.sleep(0.06)
            else:
                time.sleep(0.15)

    return Response(stream_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)