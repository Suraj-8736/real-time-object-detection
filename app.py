from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("yolo11n.pt")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "frame" not in request.files:
        return jsonify({"error": "No frame received"}), 400

    file = request.files["frame"]

    image_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    results = model(
        frame,
        conf=0.35,
        imgsz=640,
        verbose=False
    )

    annotated_frame = results[0].plot()

    success, buffer = cv2.imencode(".jpg", annotated_frame)

    if not success:
        return jsonify({"error": "Image encoding failed"}), 500

    return buffer.tobytes(), 200, {
        "Content-Type": "image/jpeg"
    }


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )