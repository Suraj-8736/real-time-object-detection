from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolo11s.pt")

# Open webcam
camera = cv2.VideoCapture(0)


def generate_frames():
    while True:

        success, frame = camera.read()

        if not success:
            break

        # Object detection
        results = model(
            frame,
            conf=0.35,
            imgsz=640,
            verbose=False
        )

        # Draw bounding boxes
        frame = results[0].plot()

        # Encode frame as JPEG
        ret, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True)