# CryFusion Project 🍼

A real-time baby cry analysis and health monitoring system using AI/ML, WebSockets, and modern web technologies.

## ✨ Features

- **🎵 Real-time Cry Analysis**: CNN-LSTM model for instant baby cry classification
- **📊 Live Health Monitoring**: Real-time sensor data with animated charts
- **🤖 AI Chat Assistant**: Gemini-powered baby care guidance
- **📱 Responsive Dashboard**: Live statistics and cry frequency tracking
- **🔴 Live Grad-CAM**: Real-time heatmap visualization during recording
- **⚡ WebSocket Integration**: Instant updates across all components

## 🏗️ Architecture

```
Frontend (React + Bootstrap) ←→ Backend (FastAPI + WebSockets) ←→ ML Model (CNN-LSTM)
                ↓                           ↓
        Real-time UI Updates         Audio Processing + Predictions
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Node.js 18+ (for development)

### 1. Clone & Setup
```bash
git clone <your-repo-url>
cd CryFusion-Project
```

### 2. Deploy with Docker
```bash
# Windows
deploy.bat

# Linux/Mac
chmod +x deploy.sh
./deploy.sh

# Manual deployment
docker-compose up --build -d
```

### 3. Access Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MongoDB**: localhost:27017

## 🔧 Development Setup

### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Frontend
```bash
cd frontend
npm install
npm start
```

## 📁 Project Structure

```
CryFusion Project/
├── backend/                 # FastAPI backend
│   ├── routes/             # API endpoints
│   ├── models/             # ML models
│   ├── data/               # Dataset
│   └── main.py            # Server entry point
├── frontend/               # React frontend
│   ├── pages/             # HTML pages
│   ├── js/                # JavaScript files
│   └── assets/            # CSS & images
├── docker-compose.yml      # Multi-service deployment
└── deploy.sh/.bat         # Deployment scripts
```

## 🎯 API Endpoints

### Core APIs
- `POST /api/cry_analysis/predict` - Audio file analysis
- `GET /api/health/sensors` - Health sensor data
- `GET /api/dashboard/stats` - Dashboard statistics
- `GET /api/dashboard/trends` - Cry frequency trends

### WebSocket Endpoints
- `ws://localhost:8000/ws/audio` - Live audio streaming
- `ws://localhost:8000/ws/sensor` - Real-time sensor updates
- `ws://localhost:8000/ws/dashboard` - Live dashboard updates

## 🧠 ML Model

- **Architecture**: CNN-LSTM hybrid
- **Input**: MFCC features (40 coefficients, 300 frames)
- **Classes**: 5 cry types (hungry, tired, discomfort, belly_pain, burping)
- **Performance**: Real-time inference (<100ms)

## 🚀 Deployment Options

### 1. Docker Compose (Recommended)
```bash
docker-compose up --build -d
```

### 2. Production Deployment
```bash
# Build production images
docker build -t cryfusion-backend ./backend
docker build -t cryfusion-frontend ./frontend

# Run with production settings
docker run -d -p 8000:8000 cryfusion-backend
docker run -d -p 3000:80 cryfusion-frontend
```

### 3. Cloud Deployment
- **AWS**: Use ECS/EKS with ALB
- **Azure**: Azure Container Instances
- **GCP**: Cloud Run or GKE
- **Heroku**: Container deployment

## 📊 Monitoring & Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Check service status
docker-compose ps

# Monitor resource usage
docker stats
```

## 🔒 Security Features

- CORS configuration for cross-origin requests
- Input validation and sanitization
- Secure WebSocket connections
- Health check endpoints
- Rate limiting (configurable)

## 🧪 Testing

```bash
# Backend tests
cd backend
python -m pytest

# Frontend tests
cd frontend
npm test

# Integration tests
docker-compose exec backend python -m pytest tests/
```

## 📈 Performance

- **Backend**: FastAPI with async/await
- **Frontend**: Optimized bundle with lazy loading
- **WebSockets**: Efficient real-time communication
- **ML Inference**: GPU acceleration support (optional)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Baby Cry Sence Dataset contributors
- FastAPI and React communities
- TensorFlow/Keras for ML framework
- Bootstrap for UI components

---

**Made with ❤️ for better baby care through AI technology**
