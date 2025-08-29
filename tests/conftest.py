#!/usr/bin/env python3
"""
🏥 Health Insurance Claim Prediction - Test Configuration

Pytest configuration and shared fixtures for all tests.

Author: Data Science Team
Date: 2024
Repository: https://github.com/odinruiz52/Health_Claim_Prediction
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Configure warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture(scope="session")  
def sample_healthcare_data():
    """
    Create realistic healthcare sample data for testing.
    
    This fixture provides consistent test data across all test modules.
    """
    np.random.seed(42)  # Reproducible data
    
    n_samples = 50
    
    # Generate realistic healthcare data
    ages = np.random.choice([
        *range(18, 30),  # Young adults
        *range(30, 50),  # Middle age
        *range(50, 70),  # Seniors
        *range(70, 85)   # Elderly
    ], n_samples)
    
    # Sex distribution (roughly equal)
    sex = np.random.choice([0, 1], n_samples, p=[0.52, 0.48])
    
    # BMI with realistic distribution
    bmi = np.random.normal(26, 4, n_samples)
    bmi = np.clip(bmi, 16, 45)  # Realistic range
    
    # Children (younger people more likely to have children)
    children = np.random.poisson(1.2, n_samples)
    children = np.clip(children, 0, 8)
    
    # Smoker (lower rate, correlated with age)
    smoker_prob = 0.15 + (ages - 18) / 200  # Slightly higher for older
    smoker = np.random.binomial(1, np.clip(smoker_prob, 0, 0.3), n_samples)
    
    # Region (uniform)
    region = np.random.choice([0, 1, 2, 3], n_samples)
    
    # Charges (correlated with age, BMI, smoking)
    base_charges = 3000 + ages * 50 + (bmi - 22) * 100
    smoker_multiplier = 1 + smoker * 1.5
    charges = base_charges * smoker_multiplier + np.random.normal(0, 1000, n_samples)
    charges = np.clip(charges, 1000, 60000)
    
    return pd.DataFrame({
        'age': ages.astype(int),
        'sex': sex.astype(int),
        'bmi': np.round(bmi, 1),
        'children': children.astype(int),
        'smoker': smoker.astype(int),
        'region': region.astype(int),
        'charges': np.round(charges, 2)
    })


@pytest.fixture(scope="session")
def sample_healthcare_target():
    """
    Create realistic target variable for healthcare data.
    
    Claim approval/denial based on realistic factors.
    """
    np.random.seed(42)
    
    # Simple rule-based target generation
    # Higher charges, smoking, older age increase denial probability
    sample_data = sample_healthcare_data()
    
    denial_prob = (
        0.2 +  # Base denial rate
        0.3 * sample_data['smoker'] +  # Smokers more likely denied
        0.2 * (sample_data['age'] > 60).astype(int) +  # Elderly more likely denied
        0.1 * (sample_data['charges'] > sample_data['charges'].quantile(0.75)).astype(int)  # High cost
    )
    
    # Add some randomness
    denial_prob = np.clip(denial_prob + np.random.normal(0, 0.1, len(sample_data)), 0.05, 0.95)
    
    # Generate binary target
    target = np.random.binomial(1, denial_prob, len(sample_data))
    
    return pd.Series(target)


@pytest.fixture
def small_sample_data():
    """Small dataset for quick tests."""
    return pd.DataFrame({
        'age': [25, 45, 65],
        'sex': [0, 1, 1],
        'bmi': [22.5, 28.0, 35.5],
        'children': [1, 2, 0],
        'smoker': [0, 1, 0],
        'region': [1, 2, 0],
        'charges': [2500.0, 8500.0, 12000.0]
    })


@pytest.fixture
def small_sample_target():
    """Small target for quick tests."""
    return pd.Series([0, 1, 1])


@pytest.fixture
def invalid_data_samples():
    """Various types of invalid data for testing validation."""
    return {
        'missing_columns': pd.DataFrame({
            'age': [25, 45],
            'sex': [0, 1]
            # Missing other required columns
        }),
        
        'null_values': pd.DataFrame({
            'age': [25, None, 65],
            'sex': [0, 1, 1], 
            'bmi': [22.5, 28.0, 35.5],
            'children': [1, 2, 0],
            'smoker': [0, 1, 0],
            'region': [1, 2, 0]
        }),
        
        'out_of_range_age': pd.DataFrame({
            'age': [200, 45, 65],  # 200 is invalid
            'sex': [0, 1, 1],
            'bmi': [22.5, 28.0, 35.5],
            'children': [1, 2, 0],
            'smoker': [0, 1, 0],
            'region': [1, 2, 0]
        }),
        
        'invalid_categorical': pd.DataFrame({
            'age': [25, 45, 65],
            'sex': [0, 1, 9],  # 9 is invalid
            'bmi': [22.5, 28.0, 35.5],
            'children': [1, 2, 0],
            'smoker': [0, 1, 0],
            'region': [1, 2, 0]
        })
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location/name."""
    for item in items:
        # Mark integration tests
        if "integration" in item.name.lower() or "end_to_end" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if any(word in item.name.lower() for word in ["slow", "large", "full"]):
            item.add_marker(pytest.mark.slow)
        
        # Default to unit test
        if not any(marker.name in ["integration", "slow"] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)