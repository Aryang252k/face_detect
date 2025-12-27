import cv2
from ultralytics import YOLO

model = YOLO("models/best.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera failed to open")
    exit()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Frame grab failed")
        break

    # DO NOT force resize first time
    frame_disp = frame.copy()

    # YOLO inference (CPU safe)
    results = model(
        frame,
        conf=0.25,
        imgsz=640,
        device="cpu",
        verbose=False
    )

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(
                frame_disp,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame_disp,
                f"Face {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("YOLO Face Stream", frame_disp)

    # IMPORTANT: allow UI refresh
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
