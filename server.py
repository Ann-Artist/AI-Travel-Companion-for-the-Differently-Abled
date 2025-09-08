from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import pyttsx3
from ultralytics import YOLO
import threading
import uvicorn

app = FastAPI()

# CORS for frontend (Next.js on 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
model = YOLO("yolov8s.pt")

# TTS engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

running = False

def run_camera():
    global running
    cap = cv2.VideoCapture(0)
    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, stream=True)
        detected_objects = set()

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                detected_objects.add(label)

        if detected_objects:
            for obj in detected_objects:
                engine.say(f"{obj} ahead")
            engine.runAndWait()

    cap.release()

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/start")
def start_detection():
    global running
    if not running:
        running = True
        threading.Thread(target=run_camera, daemon=True).start()
        return JSONResponse(content={"message": "Camera started"})
    return JSONResponse(content={"message": "Already running"})

@app.get("/stop")
def stop_detection():
    global running
    running = False
    return JSONResponse(content={"message": "Camera stopped"})

if __name__ == "__main__":
    uvicorn.run("server:app", host="172.26.215.77", port=8000, reload=True)

