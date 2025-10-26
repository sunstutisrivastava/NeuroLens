import cv2
import torch
import torchaudio
import numpy as np
import sounddevice as sd
import queue
import torch.nn.functional as F
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ======================
# Setup App
# ======================
app = FastAPI(title="NEUROLENS")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ======================
# Load Models
# ======================
image_model = torch.load("models/image_model.pth", map_location="cpu")
voice_model = torch.load("models/voice_model.pth", map_location="cpu")
image_model.eval()
voice_model.eval()

image_classes = ["angry", "disgust", "fear", "happy"]
voice_classes = ["sad", "euphoric", "joyful", "surprised"]

# ======================
# Audio Setup
# ======================
q = queue.Queue()
sr = 16000
def audio_callback(indata, frames, time, status):
    q.put(indata.copy())

sd.InputStream(callback=audio_callback, channels=1, samplerate=sr).start()

def get_audio_prediction():
    if not q.empty():
        audio = q.get().flatten()
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        mfcc = torchaudio.transforms.MFCC(n_mfcc=40)(waveform).unsqueeze(0)
        with torch.no_grad():
            out = voice_model(mfcc)
            probs = F.softmax(out, dim=1).numpy()[0]
        return {cls: float(p) for cls, p in zip(voice_classes, probs)}
    return None

def get_image_prediction():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    img = cv2.resize(frame, (128,128))
    img = torch.tensor(img, dtype=torch.float32).permute(2,0,1).unsqueeze(0) / 255.0
    with torch.no_grad():
        out = image_model(img)
        probs = F.softmax(out, dim=1).numpy()[0]
    return {cls: float(p) for cls, p in zip(image_classes, probs)}

# ======================
# Routes
# ======================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict")
async def predict():
    image_probs = get_image_prediction()
    voice_probs = get_audio_prediction()
    return JSONResponse({"image": image_probs, "voice": voice_probs})
