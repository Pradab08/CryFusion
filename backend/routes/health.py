from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import random
from datetime import datetime, timedelta
import json
import asyncio

router = APIRouter()

# Store active WebSocket connections
active_connections = []

@router.websocket("/ws/sensor")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def broadcast_sensor_update():
    """Broadcast sensor updates to all connected clients"""
    while True:
        if active_connections:
            # Generate new sensor data
            heart_rate = random.randint(90, 130)
            temperature = round(random.uniform(36.5, 38.5), 1)
            diaper_status = random.choice(["Dry", "Wet"])
            
            update_data = {
                "type": "sensor_update",
                "heart_rate": heart_rate,
                "temperature": temperature,
                "diaper_status": diaper_status,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send to all connected clients
            for connection in active_connections[:]:  # Copy list to avoid modification during iteration
                try:
                    await connection.send_text(json.dumps(update_data))
                except:
                    active_connections.remove(connection)
        
        await asyncio.sleep(2)  # Update every 2 seconds

# Start background task for sensor updates
@router.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_sensor_update())

@router.get("/api/health/sensors")
async def get_sensors():
    # Dummy current readings
    heart_rate = random.randint(90, 130)
    temperature = round(random.uniform(36.5, 38.5), 1)
    diaper_status = random.choice(["Dry", "Wet"])

    # Dummy trend data for the last 10 minutes
    now = datetime.now()
    labels = [(now - timedelta(minutes=i)).strftime("%H:%M") for i in reversed(range(10))]
    trend_hr = [random.randint(90, 130) for _ in labels]
    trend_temp = [round(random.uniform(36.5, 38.5), 1) for _ in labels]

    return {
        "heart_rate": heart_rate,
        "temperature": temperature,
        "diaper_status": diaper_status,
        "trends": {
            "time": labels,
            "heart_rate": trend_hr,
            "temperature": trend_temp,
        },
    }


