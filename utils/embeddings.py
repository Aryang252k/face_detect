import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis

detector = YOLO("./models/best.pt")

embedder = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
embedder.prepare(ctx_id=-1, det_size=(640, 640))

def get_embedding(image):
    # 1. YOLO face detection
    results = detector(image, conf=0.5)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 2. Run InsightFace on FULL IMAGE
            faces = embedder.get(image)


            if not faces:
                print("InsightFace found no faces")
                return None

            # 3. Pick the face that overlaps YOLO box
            for f in faces:
                fx1, fy1, fx2, fy2 = map(int, f.bbox)

                # simple IoU / overlap check
                if fx1 < x2 and fx2 > x1 and fy1 < y2 and fy2 > y1:
                    emb = f.embedding
                    emb = emb / np.linalg.norm(emb)
                    print("Embedding OK, shape:", emb.shape)
                    return emb.astype("float32")

    print("NO EMBEDDING GENERATED")
    return None
