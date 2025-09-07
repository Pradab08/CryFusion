import asyncio
import websockets
import soundfile as sf

async def test_ws():
    uri = "ws://localhost:8000/ws/realtime"
    async with websockets.connect(uri) as websocket:
        print("Connected to backend!")

        # Load a sample wav file
        audio, sr = sf.read(r"E:\Capstone Project\CryFusion Project\backend\data\Baby Cry Sence Dataset\burping\5afc6a14-a9d8-45f8-b31d-c79dd87cc8c6-1430757039803-1.7-m-48-bu_aug0.wav")

        # Save temporarily if you want (optional)
        sf.write("temp.wav", audio, sr)

        # Read the bytes to send
        with open("temp.wav", "rb") as f:
            audio_bytes = f.read()

        # Send audio bytes to server
        await websocket.send(audio_bytes)

        # Receive prediction
        response = await websocket.recv()
        print("Prediction:", response)

asyncio.run(test_ws())
