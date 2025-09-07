from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import io
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import motor.motor_asyncio
from pymongo import MongoClient
from pydantic import BaseModel
import base64
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    timestamp: str
    features_shape: List[int]

class HealthData(BaseModel):
    temperature: Optional[float] = None
    heart_rate: Optional[int] = None
    diaper_status: Optional[str] = None
    timestamp: str

class CryAnalysisResult(BaseModel):
    cry_analysis: PredictionResponse
    health_data: Optional[HealthData] = None
    gradcam_visualization: Optional[str] = None  # base64 encoded image

class CryFusionAPI:
    """
    FastAPI backend for CryFusion infant cry classification system
    """

    def __init__(self):
        self.app = FastAPI(
            title="CryFusion API",
            description="AI-powered infant cry classification and health monitoring system",
            version="1.0.0"
        )

        # Enable CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Initialize components
        self.model = None
        self.class_names = []
        self.gradcam_explainer = None
        self.db_client = None
        self.db = None
        self.feature_extractor_config = {
            'sample_rate': 16000,
            'n_mfcc': 13,
            'n_fft': 2048,
            'hop_length': 512,
            'max_audio_length': 5.0
        }

        # WebSocket connections for real-time updates
        self.active_connections: List[WebSocket] = []

        # Set up routes
        self._setup_routes()

    def _setup_routes(self):
        """Set up API routes"""

        @self.app.on_event("startup")
        async def startup_event():
            """Initialize models and database on startup"""
            await self.load_model()
            await self.connect_database()

        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown"""
            if self.db_client:
                self.db_client.close()

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}

        @self.app.post("/analyze_cry", response_model=PredictionResponse)
        async def analyze_cry(file: UploadFile = File(...)):
            """
            Analyze infant cry audio and return classification results
            """
            try:
                # Validate file type
                if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a')):
                    raise HTTPException(status_code=400, detail="Unsupported audio format")

                # Read audio data
                audio_data = await file.read()

                # Extract features
                features = await self.extract_features_from_audio(audio_data)

                # Make prediction
                prediction_result = await self.predict_cry_type(features)

                # Store in database
                await self.store_prediction(prediction_result, file.filename)

                # Broadcast to WebSocket clients
                await self.broadcast_prediction(prediction_result)

                return prediction_result

            except Exception as e:
                logger.error(f"Error in analyze_cry: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

        @self.app.post("/analyze_with_health", response_model=CryAnalysisResult)
        async def analyze_with_health_data(
            file: UploadFile = File(...),
            health_data: Optional[str] = None
        ):
            """
            Analyze cry with additional health sensor data
            """
            try:
                # Analyze cry
                cry_result = await self.analyze_cry_internal(file)

                # Parse health data if provided
                health_info = None
                if health_data:
                    health_info = HealthData.parse_raw(health_data)
                    await self.store_health_data(health_info)

                # Generate Grad-CAM visualization
                gradcam_viz = await self.generate_gradcam_visualization(
                    await file.read(), cry_result
                )

                result = CryAnalysisResult(
                    cry_analysis=cry_result,
                    health_data=health_info,
                    gradcam_visualization=gradcam_viz
                )

                return result

            except Exception as e:
                logger.error(f"Error in analyze_with_health: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

        @self.app.get("/history")
        async def get_analysis_history(limit: int = 50):
            """
            Get historical cry analysis results
            """
            try:
                if not self.db:
                    raise HTTPException(status_code=500, detail="Database not connected")

                cursor = self.db.cry_analyses.find().sort("timestamp", -1).limit(limit)
                history = await cursor.to_list(length=limit)

                # Convert ObjectId to string for JSON serialization
                for item in history:
                    item["_id"] = str(item["_id"])

                return {"history": history, "count": len(history)}

            except Exception as e:
                logger.error(f"Error getting history: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """
            WebSocket endpoint for real-time updates
            """
            await websocket.accept()
            self.active_connections.append(websocket)

            try:
                while True:
                    # Keep connection alive
                    await asyncio.sleep(30)
                    await websocket.send_text(json.dumps({
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat()
                    }))

            except WebSocketDisconnect:
                self.active_connections.remove(websocket)

        @self.app.get("/stats")
        async def get_system_stats():
            """
            Get system statistics and model performance metrics
            """
            try:
                if not self.db:
                    return {"error": "Database not connected"}

                # Get recent analyses count
                total_analyses = await self.db.cry_analyses.count_documents({})

                # Get class distribution
                pipeline = [
                    {"$group": {"_id": "$predicted_class", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}}
                ]
                class_distribution = await self.db.cry_analyses.aggregate(pipeline).to_list(length=None)

                # Get recent activity (last 24 hours)
                yesterday = datetime.now().timestamp() - 24 * 60 * 60
                recent_count = await self.db.cry_analyses.count_documents({
                    "timestamp": {"$gte": yesterday}
                })

                return {
                    "total_analyses": total_analyses,
                    "recent_analyses_24h": recent_count,
                    "class_distribution": class_distribution,
                    "model_loaded": self.model is not None,
                    "active_connections": len(self.active_connections)
                }

            except Exception as e:
                logger.error(f"Error getting stats: {str(e)}")
                return {"error": str(e)}

    async def load_model(self):
        """Load the trained CryFusion model"""
        try:
            model_path = "models/cryfusion_model.h5"
            config_path = "models/model_config.json"

            # Load model
            self.model = keras.models.load_model(model_path)
            logger.info("Model loaded successfully")

            # Load configuration
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.class_names = config.get('class_names', [])
            else:
                # Default class names
                self.class_names = ['belly_pain', 'burping', 'discomfort', 'hungry', 'lonely', 'scared', 'tired']

            # Initialize Grad-CAM explainer
            from evaluate_model import GradCAMExplainer
            self.gradcam_explainer = GradCAMExplainer(self.model)

            logger.info(f"Model loaded with classes: {self.class_names}")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise e

    async def connect_database(self):
        """Connect to MongoDB database"""
        try:
            # MongoDB connection string (use environment variable in production)
            mongo_url = "mongodb://localhost:27017"  # Default local MongoDB

            self.db_client = motor.motor_asyncio.AsyncIOMotorClient(mongo_url)
            self.db = self.db_client.cryfusion

            # Test connection
            await self.db.command("ping")
            logger.info("Database connected successfully")

        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            # Continue without database for development
            self.db = None

    async def extract_features_from_audio(self, audio_data: bytes) -> np.ndarray:
        """Extract MFCC features from audio data"""
        try:
            # Load audio from bytes
            audio_buffer = io.BytesIO(audio_data)
            audio, sr = librosa.load(audio_buffer, sr=self.feature_extractor_config['sample_rate'])

            # Normalize and preprocess
            audio = librosa.util.normalize(audio)
            audio, _ = librosa.effects.trim(audio, top_db=20)

            # Ensure consistent length
            target_length = int(self.feature_extractor_config['max_audio_length'] * sr)
            if len(audio) > target_length:
                audio = audio[:target_length]
            else:
                audio = np.pad(audio, (0, target_length - len(audio)), 'constant')

            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=self.feature_extractor_config['n_mfcc'],
                n_fft=self.feature_extractor_config['n_fft'],
                hop_length=self.feature_extractor_config['hop_length']
            )

            # Transpose for model input (time_steps, n_mfcc)
            features = mfcc.T

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise e

    async def predict_cry_type(self, features: np.ndarray) -> PredictionResponse:
        """Make prediction using the trained model"""
        try:
            if self.model is None:
                raise Exception("Model not loaded")

            # Add batch dimension
            features_batch = np.expand_dims(features, axis=0)

            # Make prediction
            probabilities = self.model.predict(features_batch)[0]

            # Get predicted class
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = probabilities[predicted_class_idx]

            # Create probability dictionary
            prob_dict = {
                class_name: float(prob) 
                for class_name, prob in zip(self.class_names, probabilities)
            }

            return PredictionResponse(
                predicted_class=predicted_class,
                confidence=float(confidence),
                all_probabilities=prob_dict,
                timestamp=datetime.now().isoformat(),
                features_shape=list(features.shape)
            )

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise e

    async def store_prediction(self, prediction: PredictionResponse, filename: str):
        """Store prediction result in database"""
        try:
            if self.db is None:
                return  # Skip if database not available

            document = {
                "filename": filename,
                "predicted_class": prediction.predicted_class,
                "confidence": prediction.confidence,
                "all_probabilities": prediction.all_probabilities,
                "timestamp": datetime.now().timestamp(),
                "features_shape": prediction.features_shape
            }

            await self.db.cry_analyses.insert_one(document)

        except Exception as e:
            logger.error(f"Failed to store prediction: {str(e)}")

    async def store_health_data(self, health_data: HealthData):
        """Store health sensor data"""
        try:
            if self.db is None:
                return

            document = {
                "temperature": health_data.temperature,
                "heart_rate": health_data.heart_rate,
                "diaper_status": health_data.diaper_status,
                "timestamp": datetime.now().timestamp()
            }

            await self.db.health_data.insert_one(document)

        except Exception as e:
            logger.error(f"Failed to store health data: {str(e)}")

    async def generate_gradcam_visualization(self, audio_data: bytes, prediction: PredictionResponse) -> str:
        """Generate Grad-CAM visualization and return as base64 string"""
        try:
            if self.gradcam_explainer is None:
                return None

            # Extract features
            features = await self.extract_features_from_audio(audio_data)

            # Get predicted class index
            predicted_class_idx = self.class_names.index(prediction.predicted_class)

            # Generate Grad-CAM
            heatmap = self.gradcam_explainer.generate_gradcam(features, predicted_class_idx)

            # Create visualization (simplified version)
            import matplotlib.pyplot as plt
            import io
            import base64

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(heatmap, aspect='auto', origin='lower', cmap='jet')
            ax.set_title(f'Grad-CAM: {prediction.predicted_class} (Confidence: {prediction.confidence:.2f})')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Feature Dimension')
            plt.colorbar(im)

            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return image_base64

        except Exception as e:
            logger.error(f"Grad-CAM visualization failed: {str(e)}")
            return None

    async def broadcast_prediction(self, prediction: PredictionResponse):
        """Broadcast prediction to all WebSocket connections"""
        if not self.active_connections:
            return

        message = {
            "type": "new_prediction",
            "data": prediction.dict()
        }

        # Send to all connected clients
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            self.active_connections.remove(conn)

    async def analyze_cry_internal(self, file: UploadFile) -> PredictionResponse:
        """Internal cry analysis method"""
        audio_data = await file.read()
        features = await self.extract_features_from_audio(audio_data)
        return await self.predict_cry_type(features)

# Initialize FastAPI application
cryfusion_api = CryFusionAPI()
app = cryfusion_api.app

# Run with: uvicorn fastapi_backend:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)