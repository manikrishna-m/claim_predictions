import pytest
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from src.exception import CustomException

@pytest.fixture
def sample_data():
    """
    Fixture to generate synthetic binary classification data for testing.
    Returns training and testing splits along with evaluation set.
    """
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    eval_set = [(X_test, y_test)]
    return X_train, y_train, X_test, y_test, eval_set

@pytest.fixture
def sample_config():
    """
    Fixture providing a sample configuration dictionary for the model.
    Includes XGBoost model parameters and hyperparameter tuning settings.
    """
    return {
        "model": {
            "objective": "binary:logistic",
            "n_estimators": 100,
            "random_state": 42,
            "enable_categorical": True
        },
        "tuning": {
            "param_distributions": {
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0]
            },
            "scoring": "roc_auc",
            "n_iter": 10,
            "cv": 3
        }
    }

def test_model_training_initialization(sample_config, sample_data):
    """
    Test that ModelTraining initializes correctly and stores data and configuration.
    """
    X_train, y_train, X_test, y_test, eval_set = sample_data
    
    from src.model_training import ModelTraining
    trainer = ModelTraining(sample_config, X_train, y_train, X_test, y_test, eval_set)
    
    assert trainer.X_train.shape == (80, 10)
    assert trainer.y_train.shape == (80,)
    assert trainer.X_test.shape == (20, 10)
    assert trainer.y_test.shape == (20,)
    assert len(trainer.eval_set) == 1

def test_evaluate_method(monkeypatch, sample_config, sample_data):
    """
    Test that the evaluate() method returns all expected performance metrics
    when passed a mock model implementing predict and predict_proba.
    """
    X_train, y_train, X_test, y_test, eval_set = sample_data
    
    # Define mock model
    class MockModel:
        def predict(self, X):
            return np.random.randint(0, 2, size=len(X))

        def predict_proba(self, X):
            return np.random.rand(len(X), 2)

    mock_model = MockModel()

    from src.model_training import ModelTraining
    trainer = ModelTraining(sample_config, X_train, y_train, X_test, y_test, eval_set)
    metrics = trainer.evaluate(mock_model)
    
    # Expected metric keys
    expected_metrics = [
        'train_accuracy', 'test_accuracy', 
        'train_roc_auc', 'test_roc_auc',
        'train_log_loss', 'test_log_loss',
        'f1_score', 'precision', 'recall',
        'train_kappa', 'test_kappa'
    ]
    
    assert all(metric in metrics for metric in expected_metrics)
