@echo off
echo 🚀 Deploying CryFusion Project...

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker first.
    pause
    exit /b 1
)

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not installed. Please install it first.
    pause
    exit /b 1
)

echo ✅ Docker and Docker Compose are available

REM Stop existing containers
echo 🛑 Stopping existing containers...
docker-compose down

REM Build and start services
echo 🔨 Building and starting services...
docker-compose up --build -d

REM Wait for services to be ready
echo ⏳ Waiting for services to be ready...
timeout /t 30 /nobreak >nul

REM Check service health
echo 🏥 Checking service health...

REM Check backend
curl -f http://localhost:8000/ping >nul 2>&1
if errorlevel 1 (
    echo ❌ Backend health check failed
) else (
    echo ✅ Backend is healthy
)

REM Check frontend
curl -f http://localhost:3000/health >nul 2>&1
if errorlevel 1 (
    echo ❌ Frontend health check failed
) else (
    echo ✅ Frontend is healthy
)

echo.
echo 🎉 Deployment completed!
echo.
echo 📱 Access your application:
echo    Frontend: http://localhost:3000
echo    Backend API: http://localhost:8000
echo    API Docs: http://localhost:8000/docs
echo    MongoDB: localhost:27017
echo.
echo 🔧 Useful commands:
echo    View logs: docker-compose logs -f
echo    Stop services: docker-compose down
echo    Restart services: docker-compose restart
echo    Update services: docker-compose pull ^&^& docker-compose up -d
echo.
echo 📊 Monitor services:
echo    docker-compose ps
echo    docker stats
echo.
pause
