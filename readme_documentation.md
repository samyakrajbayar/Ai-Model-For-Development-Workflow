# AI/ML Development Framework

A complete, production-ready AI/ML system for sentiment analysis with comprehensive deployment options, monitoring, and scalability features.

## 🎯 Overview

This framework demonstrates best practices in AI/ML development including:
- **Data Processing**: Automated cleaning, preprocessing, and augmentation
- **Model Training**: Multiple algorithms with hyperparameter tuning
- **Evaluation**: Comprehensive metrics and visualization
- **Deployment**: REST API with Flask/FastAPI, Docker, Kubernetes support
- **Monitoring**: Performance tracking and model monitoring
- **Scalability**: Production-ready architecture with load balancing

## 📋 Features

### Core ML Capabilities
- ✅ **Multi-algorithm Support**: Logistic Regression, Random Forest, SVM
- ✅ **Automated Hyperparameter Tuning**: Grid search with cross-validation
- ✅ **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC
- ✅ **Data Preprocessing**: Text cleaning, TF-IDF vectorization
- ✅ **Model Persistence**: Joblib serialization with metadata

### Deployment Options
- ✅ **REST APIs**: Flask and FastAPI implementations
- ✅ **Docker Support**: Multi-stage builds with health checks
- ✅ **Kubernetes**: Production-ready manifests with scaling
- ✅ **Monitoring Dashboard**: Real-time metrics and testing interface
- ✅ **Performance Testing**: Load testing and benchmarking

### Production Features
- ✅ **Error Handling**: Comprehensive exception management
- ✅ **Logging**: Structured logging with rotation
- ✅ **Health Checks**: Kubernetes/Docker compatible
- ✅ **CORS Support**: Cross-origin resource sharing
- ✅ **API Documentation**: Auto-generated OpenAPI/Swagger docs

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or create the project
git clone <your-repo> && cd ai-ml-framework

# Install dependencies
pip install -r requirements.txt

# Or create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Basic Usage

```python
# Run the complete pipeline
from ai_model import AIModelPipeline

pipeline = AIModelPipeline()
results = pipeline.run_complete_pipeline()

# View results
print(results['report'])
```

### 3. Start API Server

```bash
# Option 1: Flask (Simple)
python -c "from ai_model import start_api; start_api()"

# Option 2: FastAPI (Production)
python deployment_utilities.py --start-server

# Option 3: Using uvicorn directly
uvicorn deployment_utilities:app --host 0.0.0.0 --port 8000
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'

# Batch predictions
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '["Great product!", "Terrible service", "Average quality"]'
```

## 📁 Project Structure

```
ai-ml-framework/
├── ai_model.py                 # Main ML pipeline
├── deployment_utilities.py     # Advanced deployment features
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── models/                   # Trained models directory
│   ├── sentiment_model.pkl
│   ├── label_encoder.pkl
│   └── model_info.json
├── plots/                    # Evaluation plots
│   └── confusion_matrix.png
├── static/                   # Dashboard assets
│   └── dashboard.html
├── logs/                     # Application logs
├── k8s/                      # Kubernetes manifests
│   └── deployment.yaml
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Multi-container setup
└── .dockerignore           # Docker ignore file
```

## 🛠️ Development Workflow

### 1. Data Preparation

The framework includes a sample dataset, but you can easily integrate your own:

```python
from ai_model import DataProcessor

processor = DataProcessor()

# Load your own data
df = pd.read_csv('your_data.csv')  # Must have 'text' and 'sentiment' columns

# Or connect to database
# df = pd.read_sql('SELECT text, sentiment FROM your_table', connection)

# Preprocess
X_train, X_test, y_train, y_test = processor.prepare_data(df)
```

### 2. Model Customization

Add your own models to the training pipeline:

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# In ModelTrainer.create_baseline_models()
models['naive_bayes'] = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
    ('classifier', MultinomialNB())
])
```

### 3. Custom Evaluation Metrics

Extend the evaluation framework:

```python
from sklearn.metrics import matthews_corrcoef

# In ModelEvaluator.evaluate_model()
metrics['mcc'] = matthews_corrcoef(y_test, y_pred)
```

### 4. Advanced Preprocessing

Add custom text preprocessing:

```python
import nltk
from nltk.stem import WordNetLemmatizer

class AdvancedDataProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        # Custom preprocessing
        text = super().clean_text(text)
        # Add lemmatization, entity removal, etc.
        return text
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Create deployment files
python deployment_utilities.py --create-files

# Build Docker image
docker build -t ai-model:latest .

# Run container
docker run -p 8000:8000 ai-model:latest

# Or use Docker Compose (includes MongoDB, Redis)
docker-compose up -d
```

### Docker Compose Services

- **ai-model-api**: Main application
- **mongodb**: Prediction logging (optional)
- **redis**: Caching layer (optional)

## ☸️ Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/deployment.yaml

# Check deployment status
kubectl get deployments
kubectl get services
kubectl get pods

# Access the service
kubectl port-forward service/ai-model-service 8080:80
```

### Scaling

```bash
# Scale horizontally
kubectl scale deployment ai-model-deployment --replicas=5

# Check resource usage
kubectl top pods
```

## 📊 Monitoring and Observability

### Built-in Dashboard

Access the monitoring dashboard at: `http://localhost:8000/dashboard`

Features:
- Real-time system metrics
- Model performance statistics
- Interactive prediction testing
- Model information display

### API Endpoints

- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /stats` - Performance metrics
- `GET /model/info` - Model metadata
- `GET /dashboard` - Web dashboard

### Performance Testing

```bash
# Run built-in load tests
python deployment_utilities.py --performance-test

# Custom testing with curl
for i in {1..100}; do
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "Test message"}' &
done
wait
```

## 🔧 Configuration

### Environment Variables

```bash
export MODEL_PATH="/app/models"
export LOG_LEVEL="INFO"
export API_PORT="8000"
export WORKERS=4
```

### Production Settings

For production deployment, consider:

1. **Security**: Add API authentication
2. **Rate Limiting**: Implement request throttling
3. **SSL/TLS**: Use HTTPS certificates
4. **Database**: Persistent prediction logging
5. **Caching**: Redis for frequent predictions
6. **Monitoring**: Prometheus/Grafana integration

## 📈 Model Performance

### Expected Metrics (Sample Data)

| Metric | Value |
|--------|--------|
| Accuracy | ~85-90% |
| F1-Score | ~85-90% |
| Precision | ~85-90% |
| Recall | ~85-90% |
| Response Time | <50ms |

### Optimization Tips

1. **Feature Engineering**: 
   - N-grams (1,2) for better context
   - TF-IDF with different parameters
   - Word embeddings (Word2Vec, GloVe)

2. **Model Selection**:
   - Deep learning with transformers (BERT, RoBERTa)
   - Ensemble methods
   - Active learning for continuous improvement

3. **Performance**:
   - Model quantization
   - ONNX export for faster inference
   - Batch processing for throughput

## 🧪 Testing

### Unit Tests

```bash
# Install testing dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v --cov=ai_model
```

### Integration Tests

```bash
# Test API endpoints
python -m pytest tests/test_api.py

# Test model pipeline
python -m pytest tests/test_pipeline.py
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Check model files exist
   ls -la models/
   
   # Verify file permissions
   chmod 644 models/*.pkl
   ```

2. **API Connection Issues**
   ```bash
   # Check if port is in use
   lsof -i :8000
   
   # Test local connectivity
   curl http://localhost:8000/health
   ```

3. **Memory Issues**
   ```bash
   # Monitor memory usage
   docker stats
   
   # Adjust Docker memory limits
   docker run --memory=2g ai-model:latest
   ```

4. **Performance Issues**
   ```bash
   # Profile the application
   python -m cProfile ai_model.py
   
   # Monitor with htop/top
   htop
   ```

### Logging

Check logs for detailed error information:

```bash
# Application logs
tail -f ai_model.log

# Docker logs
docker logs <container_id>

# Kubernetes logs
kubectl logs -f deployment/ai-model-deployment
```

## 🔄 MLOps Integration

### MLflow Integration

```python
from deployment_utilities import ModelMLflowIntegration

# Initialize MLflow
mlops = ModelMLflowIntegration("sentiment_analysis")

# Log training run
mlops.log_training_run(
    model=best_model,
    metrics=evaluation_metrics,
    params=hyperparameters
)

# Load model from registry
model = mlops.load_model_from_registry("SentimentAnalysisModel", "latest")
```

### Continuous Integration

Example GitHub Actions workflow:

```yaml
name: CI/CD Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: pytest tests/
    - name: Build Docker image
      run: docker build -t ai-model:${{ github.sha }} .
```

## 📚 Advanced Usage

### Custom Model Integration

```python
import torch
from transformers import AutoModel, AutoTokenizer

class BERTSentimentModel:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def predict(self, texts):
        # Implementation for BERT predictions
        pass

# Integrate into pipeline
pipeline.trainer.models['bert'] = BERTSentimentModel()
```

### Database Integration

```python
import sqlalchemy as db

# Connect to database
engine = db.create_engine('postgresql://user:pass@localhost/dbname')

# Load data from database
query = "SELECT text, sentiment FROM reviews WHERE created_at > '2023-01-01'"
df = pd.read_sql(query, engine)

# Process and train
results = pipeline.run_complete_pipeline()
```

### Real-time Stream Processing

```python
import kafka

# Kafka consumer for real-time predictions
consumer = kafka.KafkaConsumer('text-stream')

for message in consumer:
    text = message.value.decode('utf-8')
    prediction = deployment.predict_sentiment(text)
    
    # Send to output stream
    producer.send('predictions', prediction)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- scikit-learn for machine learning algorithms
- FastAPI for modern API framework
- Docker for containerization
- Kubernetes for orchestration
- MLflow for experiment tracking

## 📞 Support

For questions, issues, or contributions:
- Create an issue in the repository
- Contact: [your-email@domain.com]
- Documentation: [link-to-docs]

---

**Happy ML Engineering! 🚀**