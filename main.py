import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open Mac webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open nahi ho rahi!")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame read nahi ho raha!")
        break

    # Object detection
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    # Show result
    cv2.imshow("Real-Time Object Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()