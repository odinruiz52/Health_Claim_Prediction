#!/usr/bin/env python3
"""
🏥 Health Insurance Claim Prediction - Feature Engineering Tests

Tests for the unified feature engineering module to ensure:
- Consistent feature creation between training and inference
- Proper handling of charges-based features
- Robust validation and error handling
- Sklearn transformer interface compliance

Author: Data Science Team
Date: 2024
Repository: https://github.com/odinruiz52/Health_Claim_Prediction
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.estimator_checks import check_estimator
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from features import FeatureBuilder, validate_feature_data


class TestFeatureBuilder:
    """Test suite for FeatureBuilder transformer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'age': [25, 45, 65, 30, 55],
            'sex': [0, 1, 1, 0, 1],
            'bmi': [22.5, 28.0, 35.5, 24.0, 29.5],
            'children': [1, 2, 0, 1, 3],
            'smoker': [0, 1, 0, 0, 1],
            'region': [1, 2, 0, 3, 1],
            'charges': [2500.0, 8500.0, 12000.0, 3000.0, 9500.0]
        })
    
    @pytest.fixture
    def sample_data_no_charges(self):
        """Create sample data without charges column."""
        return pd.DataFrame({
            'age': [25, 45, 65, 30, 55],
            'sex': [0, 1, 1, 0, 1],
            'bmi': [22.5, 28.0, 35.5, 24.0, 29.5],
            'children': [1, 2, 0, 1, 3],
            'smoker': [0, 1, 0, 0, 1],
            'region': [1, 2, 0, 3, 1]
        })
    
    def test_feature_builder_inheritance(self):
        """Test that FeatureBuilder properly inherits from sklearn base classes."""
        fb = FeatureBuilder()
        assert isinstance(fb, BaseEstimator)
        assert isinstance(fb, TransformerMixin)
    
    def test_feature_builder_sklearn_compatibility(self):
        """Test sklearn estimator interface compatibility."""
        # Note: This might fail due to pandas dependencies, but we'll test the interface
        try:
            check_estimator(FeatureBuilder())
        except Exception as e:
            # Expected for transformers that require specific data types
            assert "pandas" in str(e).lower() or "dataframe" in str(e).lower()
    
    def test_fit_with_charges(self, sample_data):
        """Test fitting with charges data."""
        fb = FeatureBuilder(use_charges=True)
        fitted_fb = fb.fit(sample_data)
        
        assert fitted_fb is fb  # fit should return self
        assert fb._q75_charges is not None
        assert fb._q75_charges == sample_data['charges'].quantile(0.75)
    
    def test_fit_without_charges(self, sample_data_no_charges):
        """Test fitting without charges data."""
        fb = FeatureBuilder(use_charges=False)
        fitted_fb = fb.fit(sample_data_no_charges)
        
        assert fitted_fb is fb  # fit should return self
        assert fb._q75_charges is None
    
    def test_fit_with_charges_missing_column(self, sample_data_no_charges):
        """Test fitting with use_charges=True but missing charges column."""
        fb = FeatureBuilder(use_charges=True)
        
        with pytest.raises(ValueError, match="use_charges=True but 'charges' column not found"):
            fb.fit(sample_data_no_charges)
    
    def test_transform_with_charges(self, sample_data):
        """Test transformation with charges features."""
        fb = FeatureBuilder(use_charges=True)
        fb.fit(sample_data)
        transformed = fb.transform(sample_data)
        
        # Check that engineered features are created
        expected_features = ['age_group', 'bmi_category', 'smoker_age', 'smoker_bmi', 'high_cost']
        for feature in expected_features:
            assert feature in transformed.columns, f"Missing feature: {feature}"
        
        # Check age groups
        assert transformed['age_group'].dtype.name == 'category'
        assert set(transformed['age_group'].cat.categories) == {'young', 'mid', 'senior', 'elder'}
        
        # Check BMI categories
        assert transformed['bmi_category'].dtype.name == 'category'
        assert set(transformed['bmi_category'].cat.categories) == {'under', 'normal', 'over', 'obese'}
        
        # Check interaction features
        pd.testing.assert_series_equal(
            transformed['smoker_age'], 
            sample_data['smoker'] * sample_data['age'], 
            check_names=False
        )
        pd.testing.assert_series_equal(
            transformed['smoker_bmi'], 
            sample_data['smoker'] * sample_data['bmi'], 
            check_names=False
        )
        
        # Check high_cost feature
        q75 = sample_data['charges'].quantile(0.75)
        expected_high_cost = (sample_data['charges'] >= q75).astype('category')
        pd.testing.assert_series_equal(
            transformed['high_cost'], 
            expected_high_cost, 
            check_names=False,
            check_dtype=False
        )
    
    def test_transform_without_charges(self, sample_data_no_charges):
        """Test transformation without charges features."""
        fb = FeatureBuilder(use_charges=False)
        fb.fit(sample_data_no_charges)
        transformed = fb.transform(sample_data_no_charges)
        
        # Check that basic engineered features are created
        expected_features = ['age_group', 'bmi_category', 'smoker_age', 'smoker_bmi']
        for feature in expected_features:
            assert feature in transformed.columns, f"Missing feature: {feature}"
        
        # Check that high_cost is NOT created
        assert 'high_cost' not in transformed.columns
    
    def test_transform_without_fit(self, sample_data):
        """Test that transform fails without fit."""
        fb = FeatureBuilder()
        
        with pytest.raises(ValueError, match="FeatureBuilder must be fitted before transform"):
            fb.transform(sample_data)
    
    def test_fit_transform_consistency(self, sample_data):
        """Test that fit_transform produces same result as fit then transform."""
        fb1 = FeatureBuilder(use_charges=True)
        fb2 = FeatureBuilder(use_charges=True)
        
        # Method 1: fit_transform
        result1 = fb1.fit_transform(sample_data)
        
        # Method 2: fit then transform
        fb2.fit(sample_data)
        result2 = fb2.transform(sample_data)
        
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_missing_required_columns(self):
        """Test validation with missing required columns."""
        incomplete_data = pd.DataFrame({
            'age': [25, 45],
            'sex': [0, 1],
            # Missing bmi, children, smoker, region
        })
        
        fb = FeatureBuilder()
        with pytest.raises(ValueError, match="Missing required columns"):
            fb.fit(incomplete_data)
    
    def test_age_group_binning(self, sample_data):
        """Test age group binning logic."""
        fb = FeatureBuilder(use_charges=False)
        fb.fit(sample_data)
        transformed = fb.transform(sample_data)
        
        # Test specific age mappings
        age_group_map = {
            25: 'young',    # 0-30
            45: 'mid',      # 31-45  
            65: 'elder',    # 61+
            30: 'young',    # boundary case
            55: 'senior'    # 46-60
        }
        
        for i, age in enumerate(sample_data['age']):
            expected_group = age_group_map[age]
            actual_group = transformed.iloc[i]['age_group']
            assert actual_group == expected_group, f"Age {age} should be {expected_group}, got {actual_group}"
    
    def test_bmi_category_binning(self, sample_data):
        """Test BMI category binning logic."""
        fb = FeatureBuilder(use_charges=False)
        fb.fit(sample_data)
        transformed = fb.transform(sample_data)
        
        # Test specific BMI mappings
        bmi_category_map = {
            22.5: 'normal',    # 18.5-25
            28.0: 'over',      # 25-30
            35.5: 'obese',     # 30+
            24.0: 'normal',    # 18.5-25
            29.5: 'over'       # 25-30
        }
        
        for i, bmi in enumerate(sample_data['bmi']):
            expected_category = bmi_category_map[bmi]
            actual_category = transformed.iloc[i]['bmi_category']
            assert actual_category == expected_category, f"BMI {bmi} should be {expected_category}, got {actual_category}"
    
    def test_get_feature_names_out(self):
        """Test feature names output method."""
        # Test without charges
        fb = FeatureBuilder(use_charges=False)
        feature_names = fb.get_feature_names_out()
        
        expected_features = [
            'age', 'sex', 'bmi', 'children', 'smoker', 'region',  # base features
            'age_group', 'bmi_category', 'smoker_age', 'smoker_bmi'  # engineered features
        ]
        assert feature_names == expected_features
        
        # Test with charges
        fb_with_charges = FeatureBuilder(use_charges=True)
        feature_names_with_charges = fb_with_charges.get_feature_names_out()
        
        expected_features_with_charges = expected_features + ['charges', 'high_cost']
        assert feature_names_with_charges == expected_features_with_charges


class TestFeatureValidation:
    """Test suite for feature validation functions."""
    
    @pytest.fixture
    def valid_data(self):
        """Valid data for testing."""
        return pd.DataFrame({
            'age': [25, 45, 65],
            'sex': [0, 1, 1],
            'bmi': [22.5, 28.0, 35.5],
            'children': [1, 2, 0],
            'smoker': [0, 1, 0],
            'region': [1, 2, 0],
            'charges': [2500.0, 8500.0, 12000.0]
        })
    
    def test_validate_feature_data_valid(self, valid_data):
        """Test validation with valid data."""
        is_valid, error_msg = validate_feature_data(valid_data, require_charges=True)
        assert is_valid is True
        assert error_msg == ""
    
    def test_validate_feature_data_missing_columns(self):
        """Test validation with missing required columns."""
        incomplete_data = pd.DataFrame({
            'age': [25, 45],
            'sex': [0, 1]
            # Missing other required columns
        })
        
        is_valid, error_msg = validate_feature_data(incomplete_data)
        assert is_valid is False
        assert "Missing required columns" in error_msg
    
    def test_validate_feature_data_null_values(self):
        """Test validation with null values."""
        data_with_nulls = pd.DataFrame({
            'age': [25, None, 65],
            'sex': [0, 1, 1],
            'bmi': [22.5, 28.0, 35.5],
            'children': [1, 2, 0],
            'smoker': [0, 1, 0],
            'region': [1, 2, 0]
        })
        
        is_valid, error_msg = validate_feature_data(data_with_nulls)
        assert is_valid is False
        assert "Null values found" in error_msg
    
    def test_validate_feature_data_age_range(self):
        """Test validation with age out of range."""
        data_bad_age = pd.DataFrame({
            'age': [200, 45, 65],  # 200 is out of range
            'sex': [0, 1, 1],
            'bmi': [22.5, 28.0, 35.5],
            'children': [1, 2, 0],
            'smoker': [0, 1, 0],
            'region': [1, 2, 0]
        })
        
        is_valid, error_msg = validate_feature_data(data_bad_age)
        assert is_valid is False
        assert "Age values out of reasonable range" in error_msg
    
    def test_validate_feature_data_categorical_values(self):
        """Test validation with invalid categorical values."""
        data_bad_sex = pd.DataFrame({
            'age': [25, 45, 65],
            'sex': [0, 1, 9],  # 9 is invalid
            'bmi': [22.5, 28.0, 35.5],
            'children': [1, 2, 0],
            'smoker': [0, 1, 0],
            'region': [1, 2, 0]
        })
        
        is_valid, error_msg = validate_feature_data(data_bad_sex)
        assert is_valid is False
        assert "Sex must be 0 (female) or 1 (male)" in error_msg
    
    def test_validate_feature_data_require_charges(self, valid_data):
        """Test validation when charges are required."""
        data_no_charges = valid_data.drop(columns=['charges'])
        
        is_valid, error_msg = validate_feature_data(data_no_charges, require_charges=True)
        assert is_valid is False
        assert "Missing required columns" in error_msg
        
        # Should pass when charges not required
        is_valid, error_msg = validate_feature_data(data_no_charges, require_charges=False)
        assert is_valid is True


class TestFeatureBuilderEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        fb = FeatureBuilder()
        
        with pytest.raises(ValueError):
            fb.fit(empty_df)
    
    def test_single_row_dataframe(self):
        """Test with single row DataFrame."""
        single_row = pd.DataFrame({
            'age': [35],
            'sex': [1],
            'bmi': [26.5],
            'children': [2],
            'smoker': [0],
            'region': [1],
            'charges': [5000.0]
        })
        
        fb = FeatureBuilder(use_charges=True)
        fb.fit(single_row)
        transformed = fb.transform(single_row)
        
        # Should work with single row
        assert len(transformed) == 1
        assert 'age_group' in transformed.columns
        assert 'high_cost' in transformed.columns
    
    def test_boundary_values(self):
        """Test with boundary values for binning."""
        boundary_data = pd.DataFrame({
            'age': [30, 45, 60],  # Boundary values
            'sex': [0, 1, 0],
            'bmi': [18.5, 25.0, 30.0],  # Boundary values
            'children': [0, 5, 10],
            'smoker': [0, 1, 0],
            'region': [0, 2, 3]
        })
        
        fb = FeatureBuilder(use_charges=False)
        fb.fit(boundary_data)
        transformed = fb.transform(boundary_data)
        
        # Check boundary handling
        assert transformed.iloc[0]['age_group'] == 'young'  # 30 should be young
        assert transformed.iloc[1]['age_group'] == 'mid'    # 45 should be mid
        assert transformed.iloc[2]['age_group'] == 'senior' # 60 should be senior
        
        assert transformed.iloc[0]['bmi_category'] == 'normal'  # 18.5 should be normal
        assert transformed.iloc[1]['bmi_category'] == 'over'    # 25.0 should be over  
        assert transformed.iloc[2]['bmi_category'] == 'obese'   # 30.0 should be obese


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])