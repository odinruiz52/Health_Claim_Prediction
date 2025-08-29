#!/usr/bin/env python3
"""
🏥 Health Insurance Claim Prediction - Feature Engineering Module

This module provides a unified feature engineering transformer that ensures
consistency between training and inference pipelines.

CRITICAL: This transformer must be used identically in both training and
inference to prevent train/serve skew that leads to incorrect predictions.

Features Created:
- age_group: Categorical age ranges (young, mid, senior, elder)
- bmi_category: BMI health categories (under, normal, over, obese)  
- smoker_age: Interaction between smoking status and age
- smoker_bmi: Interaction between smoking status and BMI
- high_cost: Optional high-cost claim indicator (training only)

Author: Data Science Team
Date: 2024
Repository: https://github.com/odinruiz52/Health_Claim_Prediction
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for healthcare-specific feature engineering.
    
    This transformer creates derived features from raw patient data while ensuring
    consistency between training and inference pipelines.
    
    Parameters:
    -----------
    use_charges : bool, default=False
        Whether to create features based on medical charges. Should be True only
        during training when charges are available. Set to False for inference
        when charges are unknown.
        
    Attributes:
    -----------
    _q75_charges : float or None
        75th percentile of charges computed during fit (if use_charges=True)
        
    Examples:
    ---------
    >>> # Training time (charges available)
    >>> fb_train = FeatureBuilder(use_charges=True)
    >>> X_train_features = fb_train.fit_transform(X_train)
    >>> 
    >>> # Inference time (charges unknown)
    >>> fb_inference = FeatureBuilder(use_charges=False)
    >>> X_new_features = fb_inference.fit_transform(X_new)
    """
    
    def __init__(self, use_charges: bool = False):
        self.use_charges = use_charges
        self._q75_charges: Optional[float] = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the feature builder to the training data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features with columns: age, sex, bmi, children, smoker, region
            If use_charges=True, must also include 'charges' column
        y : array-like, optional
            Target values (ignored, present for sklearn compatibility)
            
        Returns:
        --------
        self : FeatureBuilder
            Fitted transformer
        """
        # Validate required columns
        required_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Compute charges percentile if using charges-based features
        if self.use_charges:
            if 'charges' not in X.columns:
                raise ValueError("use_charges=True but 'charges' column not found in data")
            self._q75_charges = X['charges'].quantile(0.75)
            logger.info(f"Computed 75th percentile of charges: ${self._q75_charges:.2f}")
        else:
            self._q75_charges = None
            
        logger.info(f"FeatureBuilder fitted with use_charges={self.use_charges}")
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data by adding engineered features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features to transform
            
        Returns:
        --------
        pd.DataFrame
            Transformed data with additional engineered features
        """
        if not hasattr(self, '_q75_charges'):
            raise ValueError("FeatureBuilder must be fitted before transform")
            
        # Create copy to avoid modifying original data
        Z = X.copy()
        
        # Age Groups: Based on healthcare risk profiles
        Z['age_group'] = pd.cut(
            Z['age'], 
            bins=[0, 30, 45, 60, 120],
            labels=['young', 'mid', 'senior', 'elder'], 
            include_lowest=True,
            ordered=False  # Treat as nominal for proper encoding
        )
        
        # BMI Categories: Standard medical BMI classifications
        Z['bmi_category'] = pd.cut(
            Z['bmi'],
            bins=[0, 18.5, 25, 30, 500],
            labels=['under', 'normal', 'over', 'obese'],
            include_lowest=True,
            ordered=False  # Treat as nominal for proper encoding
        )
        
        # Interaction Features: Capture smoking risk factors
        Z['smoker_age'] = Z['smoker'] * Z['age']
        Z['smoker_bmi'] = Z['smoker'] * Z['bmi']
        
        # High-cost indicator (training only)
        if self.use_charges and 'charges' in Z.columns and self._q75_charges is not None:
            Z['high_cost'] = (Z['charges'] >= self._q75_charges).astype('category')
            logger.debug(f"Created high_cost feature with threshold ${self._q75_charges:.2f}")
        else:
            # Remove high_cost if it exists but we're not using charges
            if 'high_cost' in Z.columns:
                Z = Z.drop(columns=['high_cost'])
                
        # Log feature creation summary
        new_features = ['age_group', 'bmi_category', 'smoker_age', 'smoker_bmi']
        if self.use_charges and 'charges' in Z.columns:
            new_features.append('high_cost')
            
        logger.debug(f"Created {len(new_features)} engineered features: {new_features}")
        
        return Z
        
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformed data.
        
        Parameters:
        -----------
        input_features : array-like of str or None
            Input feature names (ignored, inferred from transform)
            
        Returns:
        --------
        list of str
            Output feature names including engineered features
        """
        base_features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        engineered_features = ['age_group', 'bmi_category', 'smoker_age', 'smoker_bmi']
        
        if self.use_charges:
            base_features.append('charges')
            engineered_features.append('high_cost')
            
        return base_features + engineered_features


def validate_feature_data(df: pd.DataFrame, require_charges: bool = False) -> tuple[bool, str]:
    """
    Validate input data for feature engineering.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data to validate
    require_charges : bool, default=False
        Whether to require 'charges' column
        
    Returns:
    --------
    tuple[bool, str]
        (is_valid, error_message)
    """
    required_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    if require_charges:
        required_cols.append('charges')
        
    # Check for missing columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
        
    # Check for null values
    null_cols = [col for col in required_cols if df[col].isnull().any()]
    if null_cols:
        return False, f"Null values found in columns: {null_cols}"
        
    # Basic range checks
    try:
        if not df['age'].between(0, 150).all():
            return False, "Age values out of reasonable range (0-150)"
        if not df['bmi'].between(10, 80).all():
            return False, "BMI values out of reasonable range (10-80)"
        if not df['children'].between(0, 20).all():
            return False, "Children count out of reasonable range (0-20)"
        if not df['sex'].isin([0, 1]).all():
            return False, "Sex must be 0 (female) or 1 (male)"
        if not df['smoker'].isin([0, 1]).all():
            return False, "Smoker must be 0 (no) or 1 (yes)"
        if not df['region'].isin([0, 1, 2, 3]).all():
            return False, "Region must be 0, 1, 2, or 3"
        if require_charges and not df['charges'].between(0, 1000000).all():
            return False, "Charges out of reasonable range (0-1M)"
    except Exception as e:
        return False, f"Data validation error: {str(e)}"
        
    return True, ""


if __name__ == "__main__":
    # Example usage and testing
    import pandas as pd
    
    print("🏥 Feature Engineering Module Test")
    print("=" * 50)
    
    # Sample data
    sample_data = pd.DataFrame({
        'age': [25, 45, 65],
        'sex': [0, 1, 1], 
        'bmi': [22.5, 28.0, 35.5],
        'children': [1, 2, 0],
        'smoker': [0, 1, 0],
        'region': [1, 2, 0],
        'charges': [2500.0, 8500.0, 12000.0]
    })
    
    print("Sample input data:")
    print(sample_data)
    
    # Test with charges (training scenario)
    fb_train = FeatureBuilder(use_charges=True)
    transformed_train = fb_train.fit_transform(sample_data)
    print("\nTransformed data (with charges):")
    print(transformed_train.dtypes)
    print(transformed_train)
    
    # Test without charges (inference scenario)  
    inference_data = sample_data.drop(columns=['charges'])
    fb_inference = FeatureBuilder(use_charges=False)
    transformed_inference = fb_inference.fit_transform(inference_data)
    print("\nTransformed data (without charges):")
    print(transformed_inference.dtypes)
    print(transformed_inference)
    
    print("\n✅ Feature engineering module test completed successfully!")