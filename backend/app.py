import json, time, os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from utils.audio_utils import decode_audio_bytes, extract_mfcc
from model_stub import predict_from_mfcc
from pymongo import MongoClient

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === MongoDB setup (adjust via env vars) ===
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://mongo:27017")
client = MongoClient(MONGO_URL)
db = client["cryfusion_db"]
sensor_col = db["sensors"]

# === Sensor model ===
class SensorIn(BaseModel):
    temperature: float
    heart_rate: int
    diaper_wet: bool

@app.post("/api/sensor")
async def receive_sensor(data: SensorIn):
    rec = data.dict()
    rec["ts"] = int(time.time() * 1000)
    sensor_col.insert_one(rec)
    # broadcast to WS consumers if needed - simplistic approach
    # For production use Redis pub/sub or internal broadcast manager.
    return {"status": "ok", "data": rec}

# === Simple in-memory list of sensor WS clients ===
sensor_clients: List[WebSocket] = []
async def broadcast_sensor(payload):
    to_remove = []
    for ws in sensor_clients:
        try:
            await ws.send_text(json.dumps(payload))
        except:
            to_remove.append(ws)
    for w in to_remove:
        sensor_clients.remove(w)

@app.websocket("/ws/sensor")
async def ws_sensor(ws: WebSocket):
    await ws.accept()
    sensor_clients.append(ws)
    try:
        while True:
            # keep connection alive; clients typically don't send
            await ws.receive_text()
    except WebSocketDisconnect:
        sensor_clients.remove(ws)

# === Audio websocket endpoint ===
@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            t0 = time.time()
            try:
                y, sr = decode_audio_bytes(data, target_sr=16000)
            except Exception as e:
                await ws.send_text(json.dumps({"type":"error","message": f"decode_failed: {str(e)}"}))
                continue
            mfcc = extract_mfcc(y, sr, n_mfcc=40)
            pred = predict_from_mfcc(mfcc)
            resp = {
                "type": "prediction",
                "label": pred["label"],
                "confidence": pred["confidence"],
                "timestamp": int(time.time() * 1000),
                "processing_ms": int((time.time() - t0) * 1000)
            }
            await ws.send_text(json.dumps(resp))
    except WebSocketDisconnect:
        print("audio client disconnected")
    except Exception as ex:
        print("audio ws error", ex)
        try:
            await ws.send_text(json.dumps({"type":"error","message":str(ex)}))
        except:
            pass

# === Simple health-check ===
@app.get("/")
async def root():
    return {"status":"ok"}
