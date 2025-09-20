"""
AI/ML Development Framework - Sentiment Analysis Model
=====================================================

A complete production-ready AI system for text sentiment classification.
Demonstrates data preprocessing, model training, evaluation, and deployment.

Author: AI/ML Engineer
Version: 1.0.0
"""

import os
import sys
import json
import pickle
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import joblib

# Deep learning imports (optional - install with pip install torch transformers)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import Trainer, TrainingArguments
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("Deep learning libraries not available. Using traditional ML only.")

# API imports for deployment
try:
    from flask import Flask, request, jsonify
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    print("API libraries not available. Install flask and fastapi for deployment.")

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_model.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data loading, cleaning, and preprocessing."""
    
    def __init__(self):
        self.vectorizer = None
        self.label_encoder = None
        
    def load_sample_data(self) -> pd.DataFrame:
        """Load sample sentiment analysis dataset."""
        # Create sample data (in production, load from CSV/database)
        sample_data = {
            'text': [
                "I love this product, it's amazing!",
                "This is terrible, worst purchase ever",
                "Great quality and fast shipping",
                "Not bad, could be better",
                "Excellent service, highly recommend",
                "Poor quality, very disappointed",
                "Good value for money",
                "Awful experience, will not buy again",
                "Perfect! Exactly what I needed",
                "Mediocre product, nothing special",
                "Outstanding customer service",
                "Complete waste of money",
                "Really impressed with the quality",
                "Not worth the price",
                "Amazing features and great design",
                "Terrible customer support",
                "Works perfectly as advertised",
                "Very poor build quality",
                "Fantastic experience overall",
                "Disappointing and overpriced"
            ],
            'sentiment': [
                'positive', 'negative', 'positive', 'neutral',
                'positive', 'negative', 'positive', 'negative',
                'positive', 'neutral', 'positive', 'negative',
                'positive', 'negative', 'positive', 'negative',
                'positive', 'negative', 'positive', 'negative'
            ]
        }
        return pd.DataFrame(sample_data)
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text data."""
        import re
        import string
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training."""
        logger.info("Preprocessing data...")
        
        # Clean text
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['sentiment'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'], y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        return X_train, X_test, y_train, y_test


class ModelTrainer:
    """Handles model training and hyperparameter tuning."""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def create_baseline_models(self) -> Dict[str, Pipeline]:
        """Create baseline ML models."""
        models = {
            'logistic_regression': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
                ('classifier', LogisticRegression(random_state=42))
            ]),
            'random_forest': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ]),
            'svm': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
                ('classifier', SVC(kernel='rbf', probability=True, random_state=42))
            ])
        }
        return models
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train multiple models and return results."""
        logger.info("Training models...")
        
        models = self.create_baseline_models()
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
            results[name] = model
            
        return results
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
        """Perform hyperparameter tuning for the best model."""
        logger.info("Performing hyperparameter tuning...")
        
        # Define parameter grid for logistic regression
        param_grid = {
            'tfidf__max_features': [500, 1000, 2000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l2']
        }
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, scoring='f1_macro', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        return self.best_model


class ModelEvaluator:
    """Handles model evaluation and metrics calculation."""
    
    def __init__(self, label_encoder: LabelEncoder):
        self.label_encoder = label_encoder
        
    def evaluate_model(self, model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        logger.info("Evaluating model...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
        }
        
        # ROC AUC for multiclass
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        except:
            metrics['roc_auc'] = 0.0
            
        return metrics, y_pred, y_pred_proba
    
    def generate_plots(self, y_test: np.ndarray, y_pred: np.ndarray, 
                      y_pred_proba: np.ndarray, save_path: str = 'plots/'):
        """Generate evaluation plots."""
        os.makedirs(save_path, exist_ok=True)
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to {save_path}")
    
    def generate_report(self, metrics: Dict[str, float], y_test: np.ndarray, 
                       y_pred: np.ndarray) -> str:
        """Generate detailed evaluation report."""
        report = f"""
Model Evaluation Report
======================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Performance Metrics:
-------------------
Accuracy:  {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall:    {metrics['recall']:.4f}
F1-Score:  {metrics['f1_score']:.4f}
ROC AUC:   {metrics['roc_auc']:.4f}

Detailed Classification Report:
------------------------------
{classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)}
"""
        return report


class ModelDeployment:
    """Handles model deployment and API creation."""
    
    def __init__(self, model: Pipeline, label_encoder: LabelEncoder):
        self.model = model
        self.label_encoder = label_encoder
        
    def save_model(self, save_path: str = 'models/'):
        """Save trained model and preprocessors."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, f'{save_path}/sentiment_model.pkl')
        joblib.dump(self.label_encoder, f'{save_path}/label_encoder.pkl')
        
        # Save model info
        model_info = {
            'model_type': type(self.model.named_steps['classifier']).__name__,
            'vectorizer_features': self.model.named_steps['tfidf'].get_feature_names_out().tolist()[:10],  # First 10 features
            'classes': self.label_encoder.classes_.tolist(),
            'created_at': datetime.now().isoformat()
        }
        
        with open(f'{save_path}/model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
            
        logger.info(f"Model saved to {save_path}")
    
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """Predict sentiment for a given text."""
        # Clean text
        cleaned_text = text.lower().translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
        
        # Predict
        prediction = self.model.predict([cleaned_text])[0]
        probabilities = self.model.predict_proba([cleaned_text])[0]
        
        # Format results
        result = {
            'text': text,
            'predicted_sentiment': self.label_encoder.inverse_transform([prediction])[0],
            'confidence': float(max(probabilities)),
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.label_encoder.classes_, probabilities)
            }
        }
        
        return result
    
    def create_flask_api(self) -> Flask:
        """Create Flask API for model serving."""
        if not API_AVAILABLE:
            raise ImportError("Flask not available. Install with: pip install flask")
            
        app = Flask(__name__)
        
        @app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})
        
        @app.route('/predict', methods=['POST'])
        def predict():
            try:
                data = request.json
                if 'text' not in data:
                    return jsonify({'error': 'Missing text field'}), 400
                
                result = self.predict_sentiment(data['text'])
                return jsonify(result)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        return app


class AIModelPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.trainer = ModelTrainer()
        self.evaluator = None
        self.deployment = None
        
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete ML pipeline."""
        logger.info("Starting AI/ML pipeline...")
        
        # 1. Data Loading and Preprocessing
        df = self.data_processor.load_sample_data()
        X_train, X_test, y_train, y_test = self.data_processor.prepare_data(df)
        
        # 2. Model Training
        models = self.trainer.train_models(X_train, y_train)
        
        # 3. Hyperparameter Tuning
        best_model = self.trainer.hyperparameter_tuning(X_train, y_train)
        
        # 4. Model Evaluation
        self.evaluator = ModelEvaluator(self.data_processor.label_encoder)
        metrics, y_pred, y_pred_proba = self.evaluator.evaluate_model(best_model, X_test, y_test)
        
        # 5. Generate Plots and Reports
        self.evaluator.generate_plots(y_test, y_pred, y_pred_proba)
        report = self.evaluator.generate_report(metrics, y_test, y_pred)
        
        # 6. Save Results
        with open('evaluation_report.txt', 'w') as f:
            f.write(report)
            
        # 7. Model Deployment Setup
        self.deployment = ModelDeployment(best_model, self.data_processor.label_encoder)
        self.deployment.save_model()
        
        logger.info("Pipeline completed successfully!")
        
        return {
            'metrics': metrics,
            'report': report,
            'model': best_model,
            'deployment': self.deployment
        }
    
    def demo_predictions(self, deployment: ModelDeployment):
        """Demonstrate model predictions."""
        test_texts = [
            "This product is absolutely fantastic!",
            "I hate this, it's completely useless",
            "It's okay, nothing special",
            "Amazing quality and great value!",
            "Worst purchase I've ever made"
        ]
        
        print("\n" + "="*50)
        print("DEMO PREDICTIONS")
        print("="*50)
        
        for text in test_texts:
            result = deployment.predict_sentiment(text)
            print(f"Text: {text}")
            print(f"Sentiment: {result['predicted_sentiment']} (confidence: {result['confidence']:.3f})")
            print("-" * 30)


def main():
    """Main execution function."""
    try:
        # Initialize pipeline
        pipeline = AIModelPipeline()
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline()
        
        # Print results
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(results['report'])
        
        # Demo predictions
        pipeline.demo_predictions(results['deployment'])
        
        # Optional: Start Flask API
        print("\n" + "="*50)
        print("DEPLOYMENT OPTIONS")
        print("="*50)
        print("1. Model saved to 'models/' directory")
        print("2. To start Flask API, run: python -c \"from ai_model import start_api; start_api()\"")
        print("3. API endpoints: GET /health, POST /predict")
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


def start_api():
    """Start the Flask API server."""
    try:
        # Load saved model
        model = joblib.load('models/sentiment_model.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        
        # Create deployment instance
        deployment = ModelDeployment(model, label_encoder)
        
        # Create and run Flask app
        app = deployment.create_flask_api()
        print("Starting Flask API server...")
        print("API available at: http://localhost:5000")
        print("Health check: GET http://localhost:5000/health")
        print("Prediction: POST http://localhost:5000/predict")
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        print(f"Failed to start API: {str(e)}")


if __name__ == "__main__":
    results = main()
