"""
Comprehensive Test Suite for AI/ML Framework
===========================================

Test coverage for all major components including data processing,
model training, evaluation, and API deployment.
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
import json
from unittest.mock import Mock, patch
import sys
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ai_model import (
        DataProcessor, ModelTrainer, ModelEvaluator, 
        ModelDeployment, AIModelPipeline
    )
    from deployment_utilities import (
        AdvancedModelDeployment, ModelMonitor, 
        ProductionUtils
    )
except ImportError as e:
    print(f"Warning: Could not import modules - {e}")
    print("Some tests may be skipped")

# Test fixtures
@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'text': [
            "I love this product!",
            "This is terrible",
            "It's okay, nothing special",
            "Amazing quality!",
            "Worst purchase ever",
            "Good value for money",
            "Poor customer service",
            "Excellent experience",
            "Not worth the price",
            "Fantastic product"
        ],
        'sentiment': [
            'positive', 'negative', 'neutral', 'positive', 'negative',
            'positive', 'negative', 'positive', 'negative', 'positive'
        ]
    })

@pytest.fixture
def data_processor():
    """Create DataProcessor instance."""
    return DataProcessor()

@pytest.fixture
def model_trainer():
    """Create ModelTrainer instance."""
    return ModelTrainer()

@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestDataProcessor:
    """Test data processing functionality."""
    
    def test_load_sample_data(self, data_processor):
        """Test sample data loading."""
        df = data_processor.load_sample_data()
        
        assert isinstance(df, pd.DataFrame)
        assert 'text' in df.columns
        assert 'sentiment' in df.columns
        assert len(df) > 0
        assert not df.isnull().any().any()
    
    def test_clean_text(self, data_processor):
        """Test text cleaning functionality."""
        dirty_text = "Hello! This is a TEST... with PUNCTUATION!!! "
        clean_text = data_processor.clean_text(dirty_text)
        
        assert clean_text == "hello this is a test with punctuation"
        assert clean_text.islower()
        assert not any(char in clean_text for char in "!.,?")
    
    def test_prepare_data(self, data_processor, sample_data):
        """Test data preparation pipeline."""
        X_train, X_test, y_train, y_test = data_processor.prepare_data(sample_data)
        
        # Check data types
        assert isinstance(X_train, pd.Series)
        assert isinstance(X_test, pd.Series)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        
        # Check data split
        total_samples = len(X_train) + len(X_test)
        assert total_samples == len(sample_data)
        assert len(X_test) / total_samples == pytest.approx(0.2, rel=0.1)
        
        # Check label encoder
        assert data_processor.label_encoder is not None
        assert len(data_processor.label_encoder.classes_) > 0


class TestModelTrainer:
    """Test model training functionality."""
    
    def test_create_baseline_models(self, model_trainer):
        """Test baseline model creation."""
        models = model_trainer.create_baseline_models()
        
        assert isinstance(models, dict)
        assert 'logistic_regression' in models
        assert 'random_forest' in models
        assert 'svm' in models
        
        for name, model in models.items():
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')
    
    def test_train_models(self, model_trainer, sample_data):
        """Test model training process."""
        # Prepare data
        processor = DataProcessor()
        X_train, _, y_train, _ = processor.prepare_data(sample_data)
        
        # Train models
        results = model_trainer.train_models(X_train, y_train)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        for name, model in results.items():
            assert hasattr(model, 'predict')
            # Test that model can make predictions
            predictions = model.predict(X_train[:2])
            assert len(predictions) == 2
    
    @pytest.mark.slow
    def test_hyperparameter_tuning(self, model_trainer, sample_data):
        """Test hyperparameter tuning (marked as slow test)."""
        processor = DataProcessor()
        X_train, _, y_train, _ = processor.prepare_data(sample_data)
        
        best_model = model_trainer.hyperparameter_tuning(X_train, y_train)
        
        assert best_model is not None
        assert hasattr(best_model, 'predict')
        assert model_trainer.best_model is not None


class TestModelEvaluator:
    """Test model evaluation functionality."""
    
    def test_evaluate_model(self, sample_data):
        """Test model evaluation metrics."""
        # Setup
        processor = DataProcessor()
        X_train, X_test, y_train, y_test = processor.prepare_data(sample_data)
        
        trainer = ModelTrainer()
        models = trainer.train_models(X_train, y_train)
        model = list(models.values())[0]  # Get first model
        
        evaluator = ModelEvaluator(processor.label_encoder)
        metrics, y_pred, y_pred_proba = evaluator.evaluate_model(model, X_test, y_test)
        
        # Check metrics
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1
        
        # Check predictions
        assert len(y_pred) == len(y_test)
        assert len(y_pred_proba) == len(y_test)
    
    def test_generate_plots(self, sample_data, temp_dir):
        """Test plot generation."""
        # Setup
        processor = DataProcessor()
        X_train, X_test, y_train, y_test = processor.prepare_data(sample_data)
        
        trainer = ModelTrainer()
        models = trainer.train_models(X_train, y_train)
        model = list(models.values())[0]
        
        evaluator = ModelEvaluator(processor.label_encoder)
        _, y_pred, y_pred_proba = evaluator.evaluate_model(model, X_test, y_test)
        
        # Generate plots
        plot_dir = os.path.join(temp_dir, 'plots')
        evaluator.generate_plots(y_test, y_pred, y_pred_proba, plot_dir)
        
        # Check if plots were created
        assert os.path.exists(plot_dir)
        assert os.path.exists(os.path.join(plot_dir, 'confusion_matrix.png'))
    
    def test_generate_report(self, sample_data):
        """Test evaluation report generation."""
        # Setup
        processor = DataProcessor()
        X_train, X_test, y_train, y_test = processor.prepare_data(sample_data)
        
        trainer = ModelTrainer()
        models = trainer.train_models(X_train, y_train)
        model = list(models.values())[0]
        
        evaluator = ModelEvaluator(processor.label_encoder)
        metrics, y_pred, y_pred_proba = evaluator.evaluate_model(model, X_test, y_test)
        
        report = evaluator.generate_report(metrics, y_test, y_pred)
        
        assert isinstance(report, str)
        assert 'Model Evaluation Report' in report
        assert 'Accuracy:' in report
        assert 'Precision:' in report


class TestModelDeployment:
    """Test model deployment functionality."""
    
    def test_save_model(self, sample_data, temp_dir):
        """Test model saving functionality."""
        # Train a model
        processor = DataProcessor()
        X_train, X_test, y_train, y_test = processor.prepare_data(sample_data)
        
        trainer = ModelTrainer()
        models = trainer.train_models(X_train, y_train)
        model = list(models.values())[0]
        
        # Save model
        deployment = ModelDeployment(model, processor.label_encoder)
        save_path = os.path.join(temp_dir, 'models')
        deployment.save_model(save_path)
        
        # Check if files were created
        assert os.path.exists(os.path.join(save_path, 'sentiment_model.pkl'))
        assert os.path.exists(os.path.join(save_path, 'label_encoder.pkl'))
        assert os.path.exists(os.path.join(save_path, 'model_info.json'))
        
        # Check model info
        with open(os.path.join(save_path, 'model_info.json'), 'r') as f:
            model_info = json.load(f)
            assert 'model_type' in model_info
            assert 'classes' in model_info
            assert 'created_at' in model_info
    
    def test_predict_sentiment(self, sample_data):
        """Test sentiment prediction."""
        # Setup
        processor = DataProcessor()
        X_train, _, y_train, _ = processor.prepare_data(sample_data)
        
        trainer = ModelTrainer()
        models = trainer.train_models(X_train, y_train)
        model = list(models.values())[0]
        
        deployment = ModelDeployment(model, processor.label_encoder)
        
        # Test prediction
        result = deployment.predict_sentiment("This is amazing!")
        
        assert 'text' in result
        assert 'predicted_sentiment' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        
        assert result['text'] == "This is amazing!"
        assert isinstance(result['confidence'], float)
        assert 0 <= result['confidence'] <= 1
        assert isinstance(result['probabilities'], dict)
    
    @pytest.mark.skipif(not hasattr(sys.modules.get('flask', None), 'Flask'), 
                       reason="Flask not available")
    def test_create_flask_api(self, sample_data):
        """Test Flask API creation."""
        # Setup
        processor = DataProcessor()
        X_train, _, y_train, _ = processor.prepare_data(sample_data)
        
        trainer = ModelTrainer()
        models = trainer.train_models(X_train, y_train)
        model = list(models.values())[0]
        
        deployment = ModelDeployment(model, processor.label_encoder)
        
        # Create Flask app
        app = deployment.create_flask_api()
        
        assert app is not None
        assert hasattr(app, 'test_client')
        
        # Test endpoints exist
        with app.test_client() as client:
            # Health check
            response = client.get('/health')
            assert response.status_code == 200
            
            # Prediction endpoint
            response = client.post('/predict', 
                                 json={'text': 'This is a test'})
            assert response.status_code == 200
            
            data = response.get_json()
            assert 'predicted_sentiment' in data


class TestAIModelPipeline:
    """Test the complete AI/ML pipeline."""
    
    def test_run_complete_pipeline(self, temp_dir):
        """Test the complete pipeline execution."""
        # Change to temp directory for this test
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            pipeline = AIModelPipeline()
            results = pipeline.run_complete_pipeline()
            
            # Check results structure
            assert 'metrics' in results
            assert 'report' in results
            assert 'model' in results
            assert 'deployment' in results
            
            # Check metrics
            metrics = results['metrics']
            expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            for metric in expected_metrics:
                assert metric in metrics
                assert isinstance(metrics[metric], (int, float))
            
            # Check that files were created
            assert os.path.exists('evaluation_report.txt')
            assert os.path.exists('models')
            assert os.path.exists('plots')
            
        finally:
            os.chdir(original_cwd)


class TestAdvancedDeployment:
    """Test advanced deployment features."""
    
    @pytest.mark.skipif('deployment_utilities' not in sys.modules, 
                       reason="deployment_utilities not available")
    def test_model_monitor(self):
        """Test model monitoring functionality."""
        monitor = ModelMonitor()
        
        # Log some predictions
        monitor.log_prediction("test text", {"sentiment": "positive"}, 50.0)
        monitor.log_prediction("another test", {"sentiment": "negative"}, 75.0)
        
        # Check stats
        stats = monitor.get_stats()
        assert stats['total_predictions'] == 2
        assert stats['average_processing_time_ms'] == 62.5
        assert 'uptime_minutes' in stats
        assert 'last_prediction_time' in stats
    
    def test_production_utils(self, temp_dir):
        """Test production utility functions."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Test file creation
            ProductionUtils.create_deployment_files()
            
            # Check if deployment files were created
            assert os.path.exists('Dockerfile')
            assert os.path.exists('docker-compose.yml')
            assert os.path.exists('k8s/deployment.yaml')
            assert os.path.exists('.dockerignore')
            
            # Test dashboard creation
            ProductionUtils.create_monitoring_dashboard()
            assert os.path.exists('static/dashboard.html')
            
            # Check dashboard content
            with open('static/dashboard.html', 'r') as f:
                content = f.read()
                assert 'AI Sentiment Analysis Dashboard' in content
                assert 'predict' in content.lower()
                
        finally:
            os.chdir(original_cwd)


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""
    
    def test_end_to_end_workflow(self, temp_dir):
        """Test complete end-to-end workflow."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # 1. Create and train model
            pipeline = AIModelPipeline()
            results = pipeline.run_complete_pipeline()
            
            # 2. Verify model was saved
            assert os.path.exists('models/sentiment_model.pkl')
            assert os.path.exists('models/label_encoder.pkl')
            
            # 3. Test predictions
            deployment = results['deployment']
            test_texts = [
                "I love this!",
                "This is terrible",
                "It's okay"
            ]
            
            for text in test_texts:
                result = deployment.predict_sentiment(text)
                assert 'predicted_sentiment' in result
                assert result['confidence'] > 0
            
            # 4. Verify evaluation artifacts
            assert os.path.exists('evaluation_report.txt')
            assert os.path.exists('plots/confusion_matrix.png')
            
        finally:
            os.chdir(original_cwd)
    
    def test_model_persistence_and_loading(self, sample_data, temp_dir):
        """Test that models can be saved and loaded correctly."""
        # Train and save model
        processor = DataProcessor()
        X_train, X_test, y_train, y_test = processor.prepare_data(sample_data)
        
        trainer = ModelTrainer()
        models = trainer.train_models(X_train, y_train)
        model = list(models.values())[0]
        
        deployment = ModelDeployment(model, processor.label_encoder)
        save_path = os.path.join(temp_dir, 'models')
        deployment.save_model(save_path)
        
        # Load model and test
        import joblib
        loaded_model = joblib.load(os.path.join(save_path, 'sentiment_model.pkl'))
        loaded_encoder = joblib.load(os.path.join(save_path, 'label_encoder.pkl'))
        
        # Test loaded model makes same predictions
        test_text = ["This is a test message"]
        original_pred = model.predict(test_text)
        loaded_pred = loaded_model.predict(test_text)
        
        assert np.array_equal(original_pred, loaded_pred)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_text_handling(self, sample_data):
        """Test handling of empty or invalid text."""
        processor = DataProcessor()
        X_train, _, y_train, _ = processor.prepare_data(sample_data)
        
        trainer = ModelTrainer()
        models = trainer.train_models(X_train, y_train)
        model = list(models.values())[0]
        
        deployment = ModelDeployment(model, processor.label_encoder)
        
        # Test with empty string
        result = deployment.predict_sentiment("")
        assert 'predicted_sentiment' in result
        
        # Test with whitespace only
        result = deployment.predict_sentiment("   ")
        assert 'predicted_sentiment' in result
    
    def test_invalid_data_handling(self):
        """Test handling of invalid input data."""
        processor = DataProcessor()
        
        # Test with missing columns
        invalid_df = pd.DataFrame({'wrong_column': ['test']})
        
        with pytest.raises((KeyError, AttributeError)):
            processor.prepare_data(invalid_df)
        
        # Test with empty dataframe
        empty_df = pd.DataFrame({'text': [], 'sentiment': []})
        
        with pytest.raises((ValueError, IndexError)):
            processor.prepare_data(empty_df)
    
    def test_model_loading_errors(self, temp_dir):
        """Test error handling when model files are missing."""
        if 'deployment_utilities' not in sys.modules:
            pytest.skip("deployment_utilities not available")
            
        # Try to load from non-existent path
        with pytest.raises((FileNotFoundError, OSError)):
            from deployment_utilities import AdvancedModelDeployment
            AdvancedModelDeployment(model_path=temp_dir)


# Performance and load testing
class TestPerformance:
    """Performance-related tests."""
    
    @pytest.mark.slow
    def test_prediction_speed(self, sample_data):
        """Test prediction speed requirements."""
        import time
        
        # Setup
        processor = DataProcessor()
        X_train, _, y_train, _ = processor.prepare_data(sample_data)
        
        trainer = ModelTrainer()
        models = trainer.train_models(X_train, y_train)
        model = list(models.values())[0]
        
        deployment = ModelDeployment(model, processor.label_encoder)
        
        # Time predictions
        test_texts = ["This is a test message"] * 100
        start_time = time.time()
        
        for text in test_texts:
            deployment.predict_sentiment(text)
        
        end_time = time.time()
        avg_time_per_prediction = (end_time - start_time) / len(test_texts)
        
        # Should be able to process predictions quickly
        assert avg_time_per_prediction < 0.1  # Less than 100ms per prediction
    
    def test_memory_usage(self, sample_data):
        """Test that memory usage is reasonable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run pipeline
        pipeline = AIModelPipeline()
        results = pipeline.run_complete_pipeline()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for this small example)
        assert memory_increase < 500


# Utility functions for testing
def run_fast_tests():
    """Run only fast tests (exclude slow ones)."""
    pytest.main([__file__, "-v", "-m", "not slow"])

def run_all_tests():
    """Run all tests including slow ones."""
    pytest.main([__file__, "-v"])

def run_specific_test(test_name):
    """Run a specific test."""
    pytest.main([__file__ + "::" + test_name, "-v"])


if __name__ == "__main__":
    # Run tests based on command line arguments
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "fast":
            run_fast_tests()
        elif sys.argv[1] == "all":
            run_all_tests()
        else:
            run_specific_test(sys.argv[1])
    else:
        run_fast_tests()

    