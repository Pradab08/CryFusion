from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Import routes
from routes import realtime_gradcam, cry
from routes import health as health_routes
from routes import dashboard as dashboard_routes

app = FastAPI(title="CryFusion Backend", version="1.0")

# CORS for local development (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(realtime_gradcam.router, prefix="/realtime", tags=["Realtime Grad-CAM"])
app.include_router(cry.router, tags=["Cry Analysis"])  # routes define full "/api/cry/..." paths
app.include_router(health_routes.router, tags=["Health"])  # routes define full "/api/health/..." paths
app.include_router(dashboard_routes.router, tags=["Dashboard"])  # routes define full "/api/dashboard/..." paths

# Root route
@app.get("/")
async def root():
    return {"message": "🚀 FastAPI server is running! Visit /docs for API docs."}

# Health check route
@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "✅ Backend is healthy"}

# Live audio streaming WebSocket: receive small WAV chunks and respond with predictions
@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        from model_tf import predict_from_audio_bytes
        while True:
            data = await websocket.receive_bytes()
            try:
                result = predict_from_audio_bytes(data)
                await websocket.send_json({
                    "type": "prediction",
                    "label": result["label"],
                    "confidence": result["confidence"],
                    "probs": result["probs"],
                })
            except Exception as ex:
                await websocket.send_json({"type": "error", "message": str(ex)})
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
