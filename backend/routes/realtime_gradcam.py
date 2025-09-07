"""from fastapi import APIRouter, WebSocket
import json

router = APIRouter()

# Dummy websocket route for now (replace with actual Grad-CAM logic later)
@router.websocket("/ws/realtime-gradcam")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive message from frontend
            data = await websocket.receive_text()

            # Dummy response (you will replace this with Grad-CAM predictions)
            response = {
                "status": "ok",
                "message": f"Received: {data}",
                "prediction": "hungry",
                "confidence": 0.92
            }

            # Send response back to frontend
            await websocket.send_text(json.dumps(response))

    except Exception as e:
        await websocket.close()
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/gradcam")
async def get_gradcam():
    return {
        "status": "success",
        "message": "Dummy Grad-CAM generated",
        "image_url": "/static/gradcam_dummy.png"
    }
