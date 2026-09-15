import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().with_name("yolo11n.pt")
model = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    global model

    if "frame" not in request.files:
        return jsonify({"error": "No frame received"}), 400

    file = request.files["frame"]

    image_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    if not MODEL_PATH.is_file():
        return jsonify({
            "error": "Model file is missing. Add yolo11n.pt to the deployment."
        }), 503

    try:
        if model is None:
            from ultralytics import YOLO

            model = YOLO(str(MODEL_PATH))

        results = model(
            frame,
            conf=0.35,
            imgsz=320,
            device="cpu",
            verbose=False
        )
    except Exception:
        app.logger.exception("Object detection failed")
        return jsonify({"error": "Object detection failed"}), 500

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
        port=int(os.environ.get("PORT") or 5001),
        debug=False
    )