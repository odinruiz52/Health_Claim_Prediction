#!/usr/bin/env python3
"""
🏥 Health Insurance Claim Prediction - ML Pipeline Builder

This module creates unified ML pipelines that prevent CV leakage and ensure
proper preprocessing for different algorithm types.

CRITICAL FIXES:
- Prevents CV leakage by putting ALL preprocessing inside pipelines
- Properly handles categorical variables with one-hot encoding for linear models
- Ensures consistent preprocessing across all model types
- Eliminates train/serve skew through unified pipeline approach

Pipeline Structure:
1. FeatureBuilder: Creates engineered features
2. ColumnTransformer: Handles scaling (numeric) and encoding (categorical)  
3. Estimator: ML algorithm (classifier or regressor)

Author: Data Science Team
Date: 2024
Repository: https://github.com/odinruiz52/Health_Claim_Prediction
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
import pandas as pd
import logging

# Core sklearn components
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# ML algorithms
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
# Local imports
from .features import FeatureBuilder

# Constants
RANDOM_STATE = 42

logger = logging.getLogger(__name__)


def build_preprocessor(task: str, use_charges_feature: bool) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer for consistent preprocessing.
    
    This transformer handles the critical distinction between numeric and
    categorical features, applying appropriate preprocessing to each type.
    
    Parameters:
    -----------
    task : str
        Either "classification" or "regression"
    use_charges_feature : bool  
        Whether to include charges-based features (training only)
        
    Returns:
    --------
    ColumnTransformer
        Configured preprocessor with proper scaling and encoding
        
    Notes:
    ------
    - Numeric features get StandardScaler (required for linear models, SVM, KNN)
    - Categorical features get OneHotEncoder (prevents ordinal assumptions)
    - handle_unknown="ignore" prevents errors on new categorical values
    """
    
    # Base columns from raw patient data
    base_numeric = ['age', 'bmi', 'children']
    base_categorical = ['sex', 'smoker', 'region']
    
    # Engineered columns added by FeatureBuilder
    engineered_numeric = ['smoker_age', 'smoker_bmi']
    engineered_categorical = ['age_group', 'bmi_category']
    
    # Build column lists based on configuration
    numeric_cols = base_numeric + engineered_numeric
    categorical_cols = base_categorical + engineered_categorical
    
    # Add charges-based features if available (training only)
    if use_charges_feature:
        numeric_cols.append('charges')
        categorical_cols.append('high_cost')
        
    logger.info(f"Preprocessor config - Numeric: {numeric_cols}")
    logger.info(f"Preprocessor config - Categorical: {categorical_cols}")
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            # StandardScaler for numeric features (mean=0, std=1)
            ('numeric', StandardScaler(), numeric_cols),
            
            # OneHotEncoder for categorical features 
            # - handle_unknown="ignore": gracefully handle new categories at inference
            # - sparse_output=False: return dense array for compatibility
            ('categorical', OneHotEncoder(
                handle_unknown='ignore', 
                sparse_output=False,
                drop='first'  # Prevent multicollinearity
            ), categorical_cols),
        ],
        remainder='drop',  # Drop any other columns
        verbose_feature_names_out=False,  # Cleaner feature names
    )
    
    return preprocessor


def build_model_pipelines(task: str, use_charges_feature: bool) -> Dict[str, Pipeline]:
    """
    Build complete ML pipelines for different algorithms.
    
    Each pipeline includes:
    1. FeatureBuilder: Consistent feature engineering
    2. Preprocessor: Algorithm-appropriate scaling/encoding  
    3. Estimator: The ML algorithm
    
    Parameters:
    -----------
    task : str
        Either "classification" or "regression"
    use_charges_feature : bool
        Whether to include charges-based features
        
    Returns:
    --------
    Dict[str, Pipeline]
        Dictionary mapping model names to complete pipelines
        
    Notes:
    ------
    All models use the same preprocessor to ensure consistent feature handling.
    This is critical for preventing the categorical encoding issues that 
    plagued linear models in the original implementation.
    """
    
    # Build shared components
    feature_builder = FeatureBuilder(use_charges=use_charges_feature)
    preprocessor = build_preprocessor(task, use_charges_feature)
    
    if task == "classification":
        models = {
            'logistic_regression': Pipeline([
                ('features', feature_builder),
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    class_weight='balanced',  # Handle class imbalance
                    solver='liblinear'  # Good for small datasets
                ))
            ]),
            
            'random_forest': Pipeline([
                ('features', feature_builder),
                ('preprocessor', preprocessor), 
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    class_weight='balanced',
                    n_jobs=-1
                ))
            ]),
            
            'svm': Pipeline([
                ('features', feature_builder),
                ('preprocessor', preprocessor),
                ('classifier', SVC(
                    kernel='rbf',
                    probability=True,  # Enable probability estimates
                    random_state=RANDOM_STATE,
                    class_weight='balanced'
                ))
            ]),
            
            'knn': Pipeline([
                ('features', feature_builder), 
                ('preprocessor', preprocessor),
                ('classifier', KNeighborsClassifier(
                    n_neighbors=5,
                    weights='distance',  # Weight by distance
                    metric='euclidean'
                ))
            ]),
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['xgboost'] = Pipeline([
                ('features', feature_builder),
                ('preprocessor', preprocessor),
                ('classifier', XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=RANDOM_STATE,
                    eval_metric='logloss',
                    verbosity=0
                ))
            ])
        else:
            logger.warning("XGBoost not available - skipping XGBoost pipeline")
            
    elif task == "regression":
        models = {
            'linear_regression': Pipeline([
                ('features', feature_builder),
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
            ]),
            
            'random_forest': Pipeline([
                ('features', feature_builder),
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                ))
            ]),
            
            'svr': Pipeline([
                ('features', feature_builder),
                ('preprocessor', preprocessor),
                ('regressor', SVR(
                    kernel='rbf',
                    gamma='scale'
                ))
            ]),
            
            'knn': Pipeline([
                ('features', feature_builder),
                ('preprocessor', preprocessor), 
                ('regressor', KNeighborsRegressor(
                    n_neighbors=5,
                    weights='distance'
                ))
            ]),
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['xgboost'] = Pipeline([
                ('features', feature_builder),
                ('preprocessor', preprocessor),
                ('regressor', XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=RANDOM_STATE,
                    verbosity=0
                ))
            ])
            
    else:
        raise ValueError(f"Invalid task: {task}. Must be 'classification' or 'regression'")
        
    logger.info(f"Built {len(models)} {task} pipelines: {list(models.keys())}")
    return models


def get_param_grids(task: str) -> Dict[str, Dict[str, List]]:
    """
    Get hyperparameter grids for grid search.
    
    Parameters:
    -----------
    task : str
        Either "classification" or "regression"
        
    Returns:
    --------
    Dict[str, Dict[str, List]]
        Hyperparameter grids for each model type
        
    Notes:
    ------
    Parameter names must match the pipeline structure:
    - Use 'classifier__param' for classification  
    - Use 'regressor__param' for regression
    - Parameters are prefixed with the pipeline step name
    """
    
    if task == "classification":
        return {
            'logistic_regression': {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__penalty': ['l1', 'l2']
            },
            
            'random_forest': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [5, 10, None],
                'classifier__min_samples_split': [2, 5, 10]
            },
            
            'svm': {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__gamma': ['scale', 'auto', 0.001, 0.01]
            },
            
            'knn': {
                'classifier__n_neighbors': [3, 5, 7, 11],
                'classifier__weights': ['uniform', 'distance']
            },
            
            'xgboost': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [3, 6, 9],
                'classifier__learning_rate': [0.01, 0.1, 0.2]
            } if XGBOOST_AVAILABLE else {}
        }
        
    else:  # regression
        return {
            'linear_regression': {},  # No hyperparameters to tune
            
            'random_forest': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [5, 10, None],
                'regressor__min_samples_split': [2, 5, 10]
            },
            
            'svr': {
                'regressor__C': [0.1, 1.0, 10.0],
                'regressor__gamma': ['scale', 'auto', 0.001, 0.01],
                'regressor__epsilon': [0.01, 0.1, 0.2]
            },
            
            'knn': {
                'regressor__n_neighbors': [3, 5, 7, 11],
                'regressor__weights': ['uniform', 'distance']
            },
            
            'xgboost': {
                'regressor__n_estimators': [50, 100, 200], 
                'regressor__max_depth': [3, 6, 9],
                'regressor__learning_rate': [0.01, 0.1, 0.2]
            } if XGBOOST_AVAILABLE else {}
        }


def validate_pipeline_input(X: pd.DataFrame, task: str, use_charges: bool = False) -> tuple[bool, str]:
    """
    Validate input data for pipeline processing.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Input data to validate
    task : str  
        Task type ("classification" or "regression")
    use_charges : bool
        Whether charges column should be present
        
    Returns:
    --------
    tuple[bool, str]
        (is_valid, error_message)
    """
    
    # Import here to avoid circular imports
    from .features import validate_feature_data
    
    # Validate basic feature data requirements
    is_valid, error_msg = validate_feature_data(X, require_charges=use_charges)
    if not is_valid:
        return False, f"Feature validation failed: {error_msg}"
    
    # Additional pipeline-specific validations
    if len(X) == 0:
        return False, "Input data is empty"
        
    if task not in ["classification", "regression"]:
        return False, f"Invalid task: {task}. Must be 'classification' or 'regression'"
        
    logger.debug(f"Pipeline input validation passed for {len(X)} samples")
    return True, ""


if __name__ == "__main__":
    # Test pipeline builder functionality
    print("🏥 Pipeline Builder Module Test")
    print("=" * 50)
    
    # Sample data for testing
    sample_data = pd.DataFrame({
        'age': [25, 45, 65, 30, 55],
        'sex': [0, 1, 1, 0, 1],
        'bmi': [22.5, 28.0, 35.5, 24.0, 29.5], 
        'children': [1, 2, 0, 1, 3],
        'smoker': [0, 1, 0, 0, 1],
        'region': [1, 2, 0, 3, 1],
        'charges': [2500.0, 8500.0, 12000.0, 3000.0, 9500.0]
    })
    
    print("Sample input data:")
    print(sample_data)
    print()
    
    # Test classification pipelines
    print("Testing classification pipelines...")
    classification_models = build_model_pipelines("classification", use_charges_feature=True)
    
    for name, pipeline in classification_models.items():
        print(f"\n{name} pipeline steps:")
        for step_name, step_obj in pipeline.steps:
            print(f"  {step_name}: {type(step_obj).__name__}")
            
        # Test pipeline fitting (quick test)
        try:
            # Create dummy target
            y_class = [0, 1, 1, 0, 1]
            pipeline_copy = pipeline
            pipeline_copy.fit(sample_data, y_class)
            predictions = pipeline_copy.predict(sample_data)
            print(f"  ✅ Pipeline test successful - predictions shape: {predictions.shape}")
        except Exception as e:
            print(f"  ❌ Pipeline test failed: {str(e)}")
    
    # Test regression pipelines
    print("\n" + "="*30)
    print("Testing regression pipelines...")
    regression_models = build_model_pipelines("regression", use_charges_feature=False)
    
    for name, pipeline in regression_models.items():
        print(f"\n{name} pipeline steps:")
        for step_name, step_obj in pipeline.steps:
            print(f"  {step_name}: {type(step_obj).__name__}")
            
    # Test parameter grids
    print("\n" + "="*30)
    print("Parameter grids:")
    class_grids = get_param_grids("classification")
    for model, params in class_grids.items():
        if params:  # Skip empty grids
            print(f"\n{model}: {list(params.keys())}")
    
    print("\n✅ Pipeline builder module test completed successfully!")