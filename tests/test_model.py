import pytest
import pandas as pd
import numpy as np
import sys
import os
from sklearn.datasets import make_classification

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator


@pytest.fixture
def sample_classification_data():
    """Create sample classification dataset"""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42
    )
    return pd.DataFrame(X), pd.Series(y)


@pytest.fixture
def config():
    """Create configuration for testing"""
    return {
        'model': {
            'handle_imbalance': True,
            'imbalance_strategy': 'smote',
            'smote_sampling_strategy': 0.5,
            'n_folds': 3
        },
        'models_to_train': [
            {
                'name': 'logistic_regression',
                'params': {'max_iter': 1000, 'class_weight': 'balanced'}
            },
            {
                'name': 'random_forest',
                'params': {'n_estimators': 50, 'max_depth': 10, 'class_weight': 'balanced'}
            }
        ],
        'hyperparameter_tuning': {
            'enable': False,
            'n_trials': 5,
            'timeout': 60
        },
        'evaluation': {
            'metrics': ['roc_auc', 'precision', 'recall', 'f1', 'pr_auc'],
            'threshold_optimization': True,
            'business_cost_fn_ratio': 5
        }
    }


def test_model_initialization(config):
    """Test model trainer initialization"""
    trainer = ModelTrainer(config)
    assert trainer.config == config
    assert isinstance(trainer.models, dict)
    assert trainer.best_model is None


def test_handle_imbalance(config, sample_classification_data):
    """Test class imbalance handling"""
    X, y = sample_classification_data
    trainer = ModelTrainer(config)
    
    X_resampled, y_resampled = trainer.handle_imbalance(X.values, y.values, strategy='smote')
    
    assert len(X_resampled) >= len(X)
    assert len(y_resampled) >= len(y)


def test_get_model(config):
    """Test model initialization by name"""
    trainer = ModelTrainer(config)
    
    lr_model = trainer.get_model('logistic_regression')
    assert lr_model is not None
    
    rf_model = trainer.get_model('random_forest')
    assert rf_model is not None


def test_train_single_model(config, sample_classification_data):
    """Test training a single model"""
    X, y = sample_classification_data
    trainer = ModelTrainer(config)
    
    model, cv_score = trainer.train_model(X, y, 'logistic_regression')
    
    assert model is not None
    assert cv_score > 0.5
    assert 'logistic_regression' in trainer.models


def test_train_all_models(config, sample_classification_data):
    """Test training all configured models"""
    X, y = sample_classification_data
    trainer = ModelTrainer(config)
    
    results = trainer.train_all_models(X, y)
    
    assert len(results) > 0
    assert trainer.best_model is not None
    assert trainer.best_model_name is not None


def test_model_evaluation(config, sample_classification_data):
    """Test model evaluation"""
    X, y = sample_classification_data
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    trainer = ModelTrainer(config)
    model, _ = trainer.train_model(X_train, y_train, 'logistic_regression')
    
    evaluator = ModelEvaluator(config)
    metrics = evaluator.evaluate_model(model, X_test, y_test, 'logistic_regression')
    
    assert 'roc_auc' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert metrics['roc_auc'] > 0.5


def test_threshold_optimization(config, sample_classification_data):
    """Test threshold optimization"""
    X, y = sample_classification_data
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    trainer = ModelTrainer(config)
    model, _ = trainer.train_model(X_train, y_train, 'logistic_regression')
    
    evaluator = ModelEvaluator(config)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    optimal_threshold, results = evaluator.optimize_threshold(y_test, y_pred_proba, cost_fn_ratio=5)
    
    assert 0 < optimal_threshold < 1
    assert len(results) > 0


def test_model_save_load(config, sample_classification_data, tmp_path):
    """Test model saving and loading"""
    X, y = sample_classification_data
    trainer = ModelTrainer(config)
    
    model, _ = trainer.train_model(X, y, 'logistic_regression')
    
    model_path = tmp_path / "test_model.pkl"
    trainer.save_model(model, str(model_path))
    
    assert model_path.exists()
    
    loaded_model = trainer.load_model(str(model_path))
    assert loaded_model is not None
    
    predictions_original = model.predict(X[:10])
    predictions_loaded = loaded_model.predict(X[:10])
    
    assert np.array_equal(predictions_original, predictions_loaded)


def test_model_comparison(config, sample_classification_data):
    """Test model comparison"""
    X, y = sample_classification_data
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    trainer = ModelTrainer(config)
    trainer.train_all_models(X_train, y_train)
    
    evaluator = ModelEvaluator(config)
    
    for model_name, model_info in trainer.models.items():
        model = model_info['model']
        evaluator.evaluate_model(model, X_test, y_test, model_name)
    
    comparison = evaluator.compare_models()
    
    assert len(comparison) > 0
    assert 'ROC-AUC' in comparison.columns


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
