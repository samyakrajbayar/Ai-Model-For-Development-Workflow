"""
Deployment Utilities and Advanced Features
==========================================

Additional utilities for production deployment, monitoring, and scaling.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List
from datetime import datetime
import asyncio

# FastAPI deployment
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Monitoring
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

import joblib
import numpy as np


class PredictionRequest(BaseModel):
    """Request model for API predictions."""
    text: str
    return_probabilities: bool = True


class PredictionResponse(BaseModel):
    """Response model for API predictions."""
    text: str
    predicted_sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float
    timestamp: str


class ModelMonitor:
    """Monitor model performance and log metrics."""
    
    def __init__(self):
        self.prediction_count = 0
        self.total_processing_time = 0
        self.predictions_log = []
        
    def log_prediction(self, request: str, response: Dict[str, Any], processing_time: float):
        """Log prediction for monitoring."""
        self.prediction_count += 1
        self.total_processing_time += processing_time
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'request': request,
            'response': response,
            'processing_time_ms': processing_time
        }
        
        self.predictions_log.append(log_entry)
        
        # Keep only last 1000 predictions in memory
        if len(self.predictions_log) > 1000:
            self.predictions_log = self.predictions_log[-1000:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        avg_processing_time = (
            self.total_processing_time / self.prediction_count 
            if self.prediction_count > 0 else 0
        )
        
        return {
            'total_predictions': self.prediction_count,
            'average_processing_time_ms': avg_processing_time,
            'uptime_minutes': time.time() // 60,  # Simplified uptime
            'last_prediction_time': (
                self.predictions_log[-1]['timestamp'] 
                if self.predictions_log else None
            )
        }


class AdvancedModelDeployment:
    """Advanced deployment with FastAPI, monitoring, and scaling."""
    
    def __init__(self, model_path: str = 'models/'):
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.monitor = ModelMonitor()
        self.load_model()
        
    def load_model(self):
        """Load the trained model and encoders."""
        try:
            self.model = joblib.load(f'{self.model_path}/sentiment_model.pkl')
            self.label_encoder = joblib.load(f'{self.model_path}/label_encoder.pkl')
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise
    
    def predict_with_timing(self, text: str) -> PredictionResponse:
        """Make prediction with timing and monitoring."""
        start_time = time.time()
        
        # Clean text
        cleaned_text = text.lower().translate(
            str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        )
        
        # Predict
        prediction = self.model.predict([cleaned_text])[0]
        probabilities = self.model.predict_proba([cleaned_text])[0]
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Format response
        response = PredictionResponse(
            text=text,
            predicted_sentiment=self.label_encoder.inverse_transform([prediction])[0],
            confidence=float(max(probabilities)),
            probabilities={
                class_name: float(prob)
                for class_name, prob in zip(self.label_encoder.classes_, probabilities)
            },
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        # Log for monitoring
        self.monitor.log_prediction(text, response.dict(), processing_time)
        
        return response
    
    def create_fastapi_app(self) -> FastAPI:
        """Create production-ready FastAPI application."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
        
        app = FastAPI(
            title="AI Sentiment Analysis API",
            description="Production-ready sentiment analysis using machine learning",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.get("/")
        async def root():
            return {"message": "AI Sentiment Analysis API", "status": "running"}
        
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "model_loaded": self.model is not None
            }
        
        @app.post("/predict", response_model=PredictionResponse)
        async def predict_sentiment(request: PredictionRequest):
            try:
                if not request.text.strip():
                    raise HTTPException(status_code=400, detail="Empty text provided")
                
                result = self.predict_with_timing(request.text)
                return result
                
            except Exception as e:
                logging.error(f"Prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        @app.post("/predict/batch")
        async def predict_batch(texts: List[str]):
            try:
                results = []
                for text in texts:
                    if text.strip():
                        result = self.predict_with_timing(text)
                        results.append(result.dict())
                    else:
                        results.append({"error": "Empty text"})
                
                return {"predictions": results, "count": len(results)}
                
            except Exception as e:
                logging.error(f"Batch prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
        
        @app.get("/stats")
        async def get_stats():
            return self.monitor.get_stats()
        
        @app.get("/model/info")
        async def model_info():
            try:
                with open(f'{self.model_path}/model_info.json', 'r') as f:
                    info = json.load(f)
                return info
            except FileNotFoundError:
                return {"error": "Model info not found"}
        
        return app


class ModelMLflowIntegration:
    """MLflow integration for experiment tracking and model registry."""
    
    def __init__(self, experiment_name: str = "sentiment_analysis"):
        if not MLFLOW_AVAILABLE:
            print("MLflow not available. Install with: pip install mlflow")
            return
            
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
    
    def log_training_run(self, model, metrics: Dict[str, float], 
                        params: Dict[str, Any]):
        """Log training run to MLflow."""
        if not MLFLOW_AVAILABLE:
            return
            
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(
                model, 
                "sentiment_model",
                registered_model_name="SentimentAnalysisModel"
            )
            
            # Log artifacts
            if os.path.exists("plots/"):
                mlflow.log_artifacts("plots/", artifact_path="evaluation_plots")
    
    def load_model_from_registry(self, model_name: str, version: str = "latest"):
        """Load model from MLflow registry."""
        if not MLFLOW_AVAILABLE:
            return None
            
        model_uri = f"models:/{model_name}/{version}"
        return mlflow.sklearn.load_model(model_uri)


# Docker configuration
DOCKERFILE_CONTENT = '''
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p models plots logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "deployment_utilities:app", "--host", "0.0.0.0", "--port", "8000"]
'''

# Docker Compose configuration
DOCKER_COMPOSE_CONTENT = '''
version: '3.8'

services:
  ai-model-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Optional: Add database for logging predictions
  mongodb:
    image: mongo:5.0
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_DATABASE: ai_model_logs
    volumes:
      - mongo_data:/data/db
    restart: unless-stopped

  # Optional: Add Redis for caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  mongo_data:
  redis_data:
'''

# Kubernetes deployment configuration
KUBERNETES_CONFIG = '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-model-deployment
  labels:
    app: ai-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-model
  template:
    metadata:
      labels:
        app: ai-model
    spec:
      containers:
      - name: ai-model-container
        image: ai-model:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/app/models"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: ai-model-service
spec:
  selector:
    app: ai-model
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
'''


class ProductionUtils:
    """Utilities for production deployment and monitoring."""
    
    @staticmethod
    def create_deployment_files():
        """Create all deployment configuration files."""
        # Create Dockerfile
        with open('Dockerfile', 'w') as f:
            f.write(DOCKERFILE_CONTENT)
        
        # Create docker-compose.yml
        with open('docker-compose.yml', 'w') as f:
            f.write(DOCKER_COMPOSE_CONTENT)
        
        # Create Kubernetes config
        os.makedirs('k8s', exist_ok=True)
        with open('k8s/deployment.yaml', 'w') as f:
            f.write(KUBERNETES_CONFIG)
        
        # Create .dockerignore
        dockerignore_content = '''
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
.git/
.gitignore
README.md
.pytest_cache/
.coverage
.DS_Store
*.log
.ipynb_checkpoints/
'''
        with open('.dockerignore', 'w') as f:
            f.write(dockerignore_content)
        
        print("Deployment files created successfully!")
        print("- Dockerfile")
        print("- docker-compose.yml") 
        print("- k8s/deployment.yaml")
        print("- .dockerignore")
    
    @staticmethod
    def create_monitoring_dashboard():
        """Create a simple monitoring dashboard."""
        dashboard_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>AI Model Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 20px 0; }
        .metric { display: inline-block; margin: 10px 20px 10px 0; }
        .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
        .metric-label { font-size: 0.9em; color: #666; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .test-area { margin-top: 20px; }
        textarea { width: 100%; height: 100px; margin: 10px 0; }
        .result { background: #f8f9fa; padding: 15px; border-radius: 4px; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Sentiment Analysis Dashboard</h1>
        
        <div class="card">
            <h2>System Status</h2>
            <div id="status">Loading...</div>
        </div>
        
        <div class="card">
            <h2>Performance Metrics</h2>
            <div id="metrics">Loading...</div>
        </div>
        
        <div class="card">
            <h2>Test the Model</h2>
            <div class="test-area">
                <textarea id="testText" placeholder="Enter text to analyze sentiment..."></textarea>
                <br>
                <button onclick="predictSentiment()">Analyze Sentiment</button>
                <div id="predictionResult"></div>
            </div>
        </div>
        
        <div class="card">
            <h2>Model Information</h2>
            <div id="modelInfo">Loading...</div>
        </div>
    </div>

    <script>
        // API base URL - adjust for your deployment
        const API_BASE = window.location.origin;
        
        // Load dashboard data
        async function loadDashboard() {
            try {
                // Load system status
                const statusResponse = await fetch(`${API_BASE}/health`);
                const statusData = await statusResponse.json();
                document.getElementById('status').innerHTML = `
                    <div class="metric">
                        <div class="metric-value">${statusData.status}</div>
                        <div class="metric-label">System Status</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${statusData.model_loaded ? 'Loaded' : 'Not Loaded'}</div>
                        <div class="metric-label">Model Status</div>
                    </div>
                `;
                
                // Load metrics
                const metricsResponse = await fetch(`${API_BASE}/stats`);
                const metricsData = await metricsResponse.json();
                document.getElementById('metrics').innerHTML = `
                    <div class="metric">
                        <div class="metric-value">${metricsData.total_predictions}</div>
                        <div class="metric-label">Total Predictions</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${Math.round(metricsData.average_processing_time_ms)}ms</div>
                        <div class="metric-label">Avg Response Time</div>
                    </div>
                `;
                
                // Load model info
                const modelResponse = await fetch(`${API_BASE}/model/info`);
                const modelData = await modelResponse.json();
                document.getElementById('modelInfo').innerHTML = `
                    <p><strong>Model Type:</strong> ${modelData.model_type || 'N/A'}</p>
                    <p><strong>Classes:</strong> ${modelData.classes ? modelData.classes.join(', ') : 'N/A'}</p>
                    <p><strong>Created:</strong> ${modelData.created_at ? new Date(modelData.created_at).toLocaleString() : 'N/A'}</p>
                `;
                
            } catch (error) {
                console.error('Error loading dashboard:', error);
            }
        }
        
        // Test model prediction
        async function predictSentiment() {
            const text = document.getElementById('testText').value;
            if (!text.trim()) {
                alert('Please enter some text to analyze');
                return;
            }
            
            try {
                const response = await fetch(`${API_BASE}/predict`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });
                
                const result = await response.json();
                
                document.getElementById('predictionResult').innerHTML = `
                    <div class="result">
                        <h3>Prediction Result</h3>
                        <p><strong>Sentiment:</strong> ${result.predicted_sentiment}</p>
                        <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                        <p><strong>Processing Time:</strong> ${result.processing_time_ms.toFixed(2)}ms</p>
                        <h4>Probabilities:</h4>
                        ${Object.entries(result.probabilities).map(([sentiment, prob]) => 
                            `<p>${sentiment}: ${(prob * 100).toFixed(1)}%</p>`
                        ).join('')}
                    </div>
                `;
                
            } catch (error) {
                document.getElementById('predictionResult').innerHTML = `
                    <div class="result" style="background: #f8d7da; color: #721c24;">
                        <strong>Error:</strong> ${error.message}
                    </div>
                `;
            }
        }
        
        // Load dashboard on page load
        loadDashboard();
        
        // Refresh metrics every 30 seconds
        setInterval(loadDashboard, 30000);
    </script>
</body>
</html>
'''
        
        os.makedirs('static', exist_ok=True)
        with open('static/dashboard.html', 'w') as f:
            f.write(dashboard_html)
        
        print("Monitoring dashboard created: static/dashboard.html")
    
    @staticmethod
    def run_performance_tests():
        """Run basic performance tests."""
        import asyncio
        import aiohttp
        import time
        
        async def test_endpoint(session, url, data):
            start_time = time.time()
            try:
                async with session.post(url, json=data) as response:
                    result = await response.json()
                    end_time = time.time()
                    return {
                        'status': response.status,
                        'time': (end_time - start_time) * 1000,
                        'success': True
                    }
            except Exception as e:
                end_time = time.time()
                return {
                    'status': 500,
                    'time': (end_time - start_time) * 1000,
                    'success': False,
                    'error': str(e)
                }
        
        async def run_load_test():
            url = "http://localhost:8000/predict"
            test_data = {"text": "This is a test message for performance testing"}
            
            async with aiohttp.ClientSession() as session:
                # Warm up
                await test_endpoint(session, url, test_data)
                
                # Run concurrent tests
                tasks = []
                for _ in range(50):  # 50 concurrent requests
                    tasks.append(test_endpoint(session, url, test_data))
                
                results = await asyncio.gather(*tasks)
                
                # Calculate statistics
                successful_requests = [r for r in results if r['success']]
                if successful_requests:
                    times = [r['time'] for r in successful_requests]
                    print(f"Load Test Results:")
                    print(f"Successful requests: {len(successful_requests)}/50")
                    print(f"Average response time: {np.mean(times):.2f}ms")
                    print(f"95th percentile: {np.percentile(times, 95):.2f}ms")
                    print(f"Max response time: {max(times):.2f}ms")
                else:
                    print("All requests failed")
        
        print("Running performance tests...")
        try:
            asyncio.run(run_load_test())
        except Exception as e:
            print(f"Performance test failed: {str(e)}")
            print("Make sure the API server is running on http://localhost:8000")


# Create FastAPI app instance for production deployment
def create_production_app():
    """Create production-ready FastAPI app."""
    deployment = AdvancedModelDeployment()
    app = deployment.create_fastapi_app()
    
    # Add static files serving for dashboard
    try:
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse
        
        if os.path.exists("static"):
            app.mount("/static", StaticFiles(directory="static"), name="static")
            
            @app.get("/dashboard")
            async def dashboard():
                return FileResponse("static/dashboard.html")
                
    except ImportError:
        pass
    
    return app


# Global app instance for uvicorn
app = None

def initialize_app():
    """Initialize the FastAPI app."""
    global app
    if app is None:
        try:
            app = create_production_app()
        except Exception as e:
            print(f"Failed to initialize app: {str(e)}")
            # Create a minimal app as fallback
            if FASTAPI_AVAILABLE:
                app = FastAPI()
                
                @app.get("/")
                async def root():
                    return {"error": "Failed to load model", "message": str(e)}
    return app


# Initialize app on import
app = initialize_app()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Model Deployment Utilities')
    parser.add_argument('--create-files', action='store_true', 
                       help='Create deployment configuration files')
    parser.add_argument('--create-dashboard', action='store_true',
                       help='Create monitoring dashboard')
    parser.add_argument('--performance-test', action='store_true',
                       help='Run performance tests')
    parser.add_argument('--start-server', action='store_true',
                       help='Start FastAPI server')
    
    args = parser.parse_args()
    
    if args.create_files:
        ProductionUtils.create_deployment_files()
    
    if args.create_dashboard:
        ProductionUtils.create_monitoring_dashboard()
    
    if args.performance_test:
        ProductionUtils.run_performance_tests()
    
    if args.start_server:
        print("Starting FastAPI server...")
        print("API Documentation: http://localhost:8000/docs")
        print("Dashboard: http://localhost:8000/dashboard")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    if not any(vars(args).values()):
        print("No action specified. Use --help for options.")