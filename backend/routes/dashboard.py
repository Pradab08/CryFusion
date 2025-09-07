from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import random
from datetime import datetime, timedelta
import json
import asyncio

router = APIRouter()

# Store active WebSocket connections
active_connections = []

@router.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def broadcast_dashboard_update():
    """Broadcast dashboard updates to all connected clients"""
    while True:
        if active_connections:
            # Generate new dashboard data
            update_data = {
                "type": "dashboard_update",
                "total_cries": random.randint(5, 25),
                "heart_rate": random.randint(90, 130),
                "temperature": round(random.uniform(36.5, 38.5), 1),
                "diaper_status": random.choice(["Dry", "Wet"]),
                "last_cry_type": random.choice(["hungry", "tired", "discomfort", "belly_pain", "burping"]),
                "last_cry_time": datetime.now().strftime("%H:%M"),
                "cry_data": {
                    "hour": datetime.now().hour,
                    "count": random.randint(0, 5)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Send to all connected clients
            for connection in active_connections[:]:
                try:
                    await connection.send_text(json.dumps(update_data))
                except:
                    active_connections.remove(connection)
        
        await asyncio.sleep(5)  # Update every 5 seconds

# Start background task for dashboard updates
@router.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_dashboard_update())

@router.get("/api/dashboard/stats")
async def get_dashboard_stats():
    return {
        "total_cries": random.randint(5, 25),
        "heart_rate": random.randint(90, 130),
        "temperature": round(random.uniform(36.5, 38.5), 1),
        "diaper_status": random.choice(["Dry", "Wet"]),
        "last_cry_type": random.choice(["hungry", "tired", "discomfort", "belly_pain", "burping"]),
        "last_cry_time": datetime.now().strftime("%H:%M"),
    }

@router.get("/api/dashboard/trends")
async def get_dashboard_trends():
    now = datetime.now()
    labels = [(now - timedelta(hours=i)).strftime("%H:%M") for i in reversed(range(12))]
    data = [random.randint(0, 5) for _ in labels]
    return {"labels": labels, "data": data}


