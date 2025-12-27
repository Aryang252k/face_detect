import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from utils.utility import add_person,update_person,search_person
from utils.embeddings import get_embedding,detector,embedder
from liveness_model import is_live
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Face Recognition API")

app.add_middleware(
    CORSMiddleware,
   allow_origins=[
        "http://127.0.0.1:5500"
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Form

@app.post("/enroll")
async def enroll(
    person_id: str = Form(...),
    file: UploadFile = File(...)
):
    image = cv2.imdecode(
        np.frombuffer(await file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    if is_live(image):
        return {"status": "Upload another image"}

    embedding = get_embedding(image)
    if embedding is None:
        return {"status": "failed"}

    add_person(embedding, person_id)
    return {"status": "enrolled", "person_id": person_id}


@app.post("/update")
async def update(
    person_id: str = Form(...),
    file: UploadFile = File(...)
):
    image = cv2.imdecode(
        np.frombuffer(await file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    embedding = get_embedding(image)
    if embedding is None:
        return {"status": "failed"}

    update_person(embedding, person_id)
    return {"status": "updated", "person_id": person_id}


import time

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = cv2.imdecode(
        np.frombuffer(img_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )


    if image is None:
        return {"person_id": None, "confidence": 0.0}
    
    if is_live(image):
        return {"status": "Upload another image"}

   
    results = detector(
        image,
        conf=0.50,
        imgsz=640,
        device="cpu",
        verbose=False
    )

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        if not hasattr(recognize, "last_embed"):
            recognize.last_embed = 0

        if time.time() - recognize.last_embed < 1.0:
            # Skip expensive InsightFace this time
            return {"person_id": None, "confidence": 0.0}

        recognize.last_embed = time.time()
        # =========================

       
        faces = embedder.get(image)
        if not faces:
            return {"person_id": None, "confidence": 0.0}

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            for f in faces:
                fx1, fy1, fx2, fy2 = map(int, f.bbox)

                if fx1 < x2 and fx2 > x1 and fy1 < y2 and fy2 > y1:
                    emb = f.embedding
                    emb = emb / np.linalg.norm(emb)
                    emb = emb.astype("float32")

                    person_id, score = search_person(emb, threshold=0.6)

                    return {
                        "person_id": person_id,
                        "confidence": score,
                        "bbox": [x1, y1, x2, y2]
                    }

    return {"person_id": None, "confidence": 0.0}


@app.get("/status")
def status():
    return {"server":"ok"}










