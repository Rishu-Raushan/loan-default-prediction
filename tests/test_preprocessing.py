import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample dataset for testing"""
    data = {
        'ID': [1, 2, 3, 4, 5],
        'Client_Income': [50000, 75000, None, 100000, 60000],
        'Credit_Amount': [10000, 15000, 20000, None, 12000],
        'Age_Days': [-10000, -15000, -20000, -12000, -18000],
        'Client_Education': ['Higher', 'Secondary', 'Higher', None, 'Secondary'],
        'Default': [0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def preprocessor():
    """Create preprocessor instance"""
    config = {
        'preprocessing': {
            'missing_value_threshold': 0.5,
            'outlier_method': 'iqr',
            'outlier_threshold': 1.5,
            'scaling_method': 'standard',
            'encoding_method': 'target'
        }
    }
    return DataPreprocessor(config)


def test_load_data(preprocessor):
    """Test data loading"""
    df = preprocessor.load_data('data/Dataset.csv')
    assert df is not None
    assert len(df) > 0
    assert df.shape[1] > 0


def test_handle_missing_values(preprocessor, sample_data):
    """Test missing value handling"""
    df_clean = preprocessor.handle_missing_values(sample_data)
    assert df_clean['Client_Income'].isnull().sum() == 0
    assert df_clean['Credit_Amount'].isnull().sum() == 0


def test_detect_outliers(preprocessor, sample_data):
    """Test outlier detection"""
    outlier_info = preprocessor.detect_outliers(sample_data)
    assert isinstance(outlier_info, dict)
    assert 'Client_Income' in outlier_info


def test_handle_outliers(preprocessor, sample_data):
    """Test outlier handling"""
    df_clean = preprocessor.handle_missing_values(sample_data)
    df_no_outliers = preprocessor.handle_outliers(df_clean)
    assert df_no_outliers is not None
    assert len(df_no_outliers) == len(df_clean)


def test_encode_categorical_features(preprocessor, sample_data):
    """Test categorical encoding"""
    df_clean = preprocessor.handle_missing_values(sample_data)
    df_encoded = preprocessor.encode_categorical_features(df_clean, target_col='Default', fit=True)
    assert df_encoded is not None
    assert 'Client_Education' not in df_encoded.columns or df_encoded['Client_Education'].dtype != 'object'


def test_scale_features(preprocessor, sample_data):
    """Test feature scaling"""
    df_clean = preprocessor.handle_missing_values(sample_data)
    df_scaled = preprocessor.scale_features(df_clean, fit=True)
    assert df_scaled is not None
    scaled_mean = df_scaled['Client_Income'].mean()
    assert abs(scaled_mean) < 1e-10


def test_preprocessing_pipeline(preprocessor, sample_data):
    """Test complete preprocessing pipeline"""
    df_processed = preprocessor.preprocess_pipeline(sample_data, target_col='Default', fit=True)
    assert df_processed is not None
    assert len(df_processed) > 0
    assert df_processed.isnull().sum().sum() == 0


def test_preprocessor_fit_transform():
    """Test preprocessor fit and transform consistency"""
    config = {
        'preprocessing': {
            'missing_value_threshold': 0.5,
            'outlier_method': 'iqr',
            'outlier_threshold': 1.5,
            'scaling_method': 'standard',
            'encoding_method': 'target'
        }
    }
    
    train_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': ['A', 'B', 'A', 'B', 'A'],
        'target': [0, 1, 0, 1, 0]
    })
    
    test_data = pd.DataFrame({
        'feature1': [2, 3, 4],
        'feature2': ['A', 'B', 'C'],
        'target': [0, 1, 0]
    })
    
    preprocessor = DataPreprocessor(config)
    
    train_processed = preprocessor.preprocess_pipeline(train_data, target_col='target', fit=True)
    test_processed = preprocessor.preprocess_pipeline(test_data, target_col='target', fit=False)
    
    assert train_processed.shape[1] >= test_processed.shape[1]
