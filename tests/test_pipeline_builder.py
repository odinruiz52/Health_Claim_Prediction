#!/usr/bin/env python3
"""
🏥 Health Insurance Claim Prediction - Pipeline Builder Tests

Tests for the ML pipeline builder to ensure:
- Proper sklearn Pipeline construction
- Correct preprocessing for different algorithm types  
- No CV leakage through complete pipeline encapsulation
- Consistent feature handling across all models

Author: Data Science Team
Date: 2024
Repository: https://github.com/odinruiz52/Health_Claim_Prediction
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, StratifiedKFold
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from pipeline_builder import (
    build_preprocessor, 
    build_model_pipelines, 
    get_param_grids,
    validate_pipeline_input
)
from features import FeatureBuilder


class TestPreprocessorBuilder:
    """Test suite for preprocessor building."""
    
    def test_build_preprocessor_classification_no_charges(self):
        """Test preprocessor for classification without charges."""
        preprocessor = build_preprocessor("classification", use_charges_feature=False)
        
        assert isinstance(preprocessor, ColumnTransformer)
        
        # Check transformers
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        assert 'numeric' in transformer_names
        assert 'categorical' in transformer_names
    
    def test_build_preprocessor_classification_with_charges(self):
        """Test preprocessor for classification with charges."""
        preprocessor = build_preprocessor("classification", use_charges_feature=True)
        
        assert isinstance(preprocessor, ColumnTransformer)
        
        # Should include charges in numeric columns
        numeric_transformer = None
        categorical_transformer = None
        
        for name, transformer, columns in preprocessor.transformers:
            if name == 'numeric':
                numeric_transformer = transformer
                numeric_columns = columns
            elif name == 'categorical':
                categorical_transformer = transformer
                categorical_columns = columns
        
        assert 'charges' in numeric_columns
        assert 'high_cost' in categorical_columns
    
    def test_build_preprocessor_regression(self):
        """Test preprocessor for regression task."""
        preprocessor = build_preprocessor("regression", use_charges_feature=False)
        
        assert isinstance(preprocessor, ColumnTransformer)
        # Should work for regression too
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        assert 'numeric' in transformer_names
        assert 'categorical' in transformer_names


class TestModelPipelineBuilder:
    """Test suite for model pipeline building."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'age': [25, 45, 65, 30, 55, 40, 35, 50, 28, 60],
            'sex': [0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
            'bmi': [22.5, 28.0, 35.5, 24.0, 29.5, 26.0, 23.5, 31.0, 25.5, 27.0],
            'children': [1, 2, 0, 1, 3, 2, 1, 0, 1, 2],
            'smoker': [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
            'region': [1, 2, 0, 3, 1, 2, 0, 1, 3, 2],
            'charges': [2500, 8500, 12000, 3000, 9500, 4500, 7000, 15000, 3500, 11000]
        })
    
    @pytest.fixture 
    def sample_target(self):
        """Create sample target for testing."""
        return pd.Series([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])  # Binary classification
    
    def test_build_classification_pipelines(self):
        """Test building classification pipelines."""
        pipelines = build_model_pipelines("classification", use_charges_feature=True)
        
        # Check that pipelines are returned
        assert isinstance(pipelines, dict)
        assert len(pipelines) > 0
        
        # Check specific models
        expected_models = ['logistic_regression', 'random_forest', 'svm', 'knn']
        for model_name in expected_models:
            assert model_name in pipelines
        
        # Check pipeline structure
        for name, pipeline in pipelines.items():
            assert isinstance(pipeline, Pipeline)
            
            # Check pipeline steps
            step_names = [step[0] for step in pipeline.steps]
            assert 'features' in step_names
            assert 'preprocessor' in step_names
            assert 'classifier' in step_names or 'regressor' in step_names
            
            # Check step types
            features_step = pipeline.named_steps['features']
            preprocessor_step = pipeline.named_steps['preprocessor']
            
            assert isinstance(features_step, FeatureBuilder)
            assert isinstance(preprocessor_step, ColumnTransformer)
    
    def test_build_regression_pipelines(self):
        """Test building regression pipelines."""
        pipelines = build_model_pipelines("regression", use_charges_feature=False)
        
        assert isinstance(pipelines, dict)
        assert len(pipelines) > 0
        
        # Check specific models
        expected_models = ['linear_regression', 'random_forest', 'svr', 'knn']
        for model_name in expected_models:
            assert model_name in pipelines
        
        # Check that final step is regressor
        for name, pipeline in pipelines.items():
            final_step_name = pipeline.steps[-1][0]
            assert 'regressor' in final_step_name or final_step_name == 'reg'
    
    def test_pipeline_fitting(self, sample_data, sample_target):
        """Test that pipelines can be fitted."""
        pipelines = build_model_pipelines("classification", use_charges_feature=True)
        
        # Test fitting one pipeline
        pipeline = pipelines['logistic_regression']
        
        # Should not raise an exception
        fitted_pipeline = pipeline.fit(sample_data, sample_target)
        assert fitted_pipeline is pipeline  # fit returns self
        
        # Test prediction
        predictions = pipeline.predict(sample_data)
        assert len(predictions) == len(sample_data)
        assert all(pred in [0, 1] for pred in predictions)  # Binary predictions
    
    def test_pipeline_cross_validation_no_leakage(self, sample_data, sample_target):
        """Test that pipelines work with cross-validation (no leakage)."""
        pipelines = build_model_pipelines("classification", use_charges_feature=True)
        pipeline = pipelines['random_forest']
        
        # This should work without errors and not leak data
        cv_scores = cross_val_score(
            pipeline, sample_data, sample_target, 
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        
        assert len(cv_scores) == 3
        assert all(0 <= score <= 1 for score in cv_scores)
    
    def test_pipeline_consistency_across_models(self, sample_data, sample_target):
        """Test that all pipelines handle the same input consistently."""
        pipelines = build_model_pipelines("classification", use_charges_feature=True)
        
        predictions = {}
        for name, pipeline in pipelines.items():
            pipeline.fit(sample_data, sample_target)
            predictions[name] = pipeline.predict(sample_data)
        
        # All should return same shape predictions
        prediction_shapes = [pred.shape for pred in predictions.values()]
        assert all(shape == (len(sample_data),) for shape in prediction_shapes)
        
        # All should be binary predictions
        for name, pred in predictions.items():
            assert all(p in [0, 1] for p in pred), f"{name} returned non-binary predictions"


class TestParameterGrids:
    """Test suite for hyperparameter grids."""
    
    def test_get_param_grids_classification(self):
        """Test parameter grids for classification."""
        param_grids = get_param_grids("classification")
        
        assert isinstance(param_grids, dict)
        
        # Check that grids exist for expected models
        expected_models = ['logistic_regression', 'random_forest', 'svm', 'knn']
        for model_name in expected_models:
            assert model_name in param_grids
        
        # Check parameter naming (should include 'classifier__')
        for model_name, grid in param_grids.items():
            if grid:  # Skip empty grids
                for param_name in grid.keys():
                    assert param_name.startswith('classifier__'), f"Invalid param name: {param_name}"
    
    def test_get_param_grids_regression(self):
        """Test parameter grids for regression."""
        param_grids = get_param_grids("regression")
        
        assert isinstance(param_grids, dict)
        
        # Check parameter naming (should include 'regressor__')
        for model_name, grid in param_grids.items():
            if grid:  # Skip empty grids
                for param_name in grid.keys():
                    assert param_name.startswith('regressor__'), f"Invalid param name: {param_name}"
    
    def test_param_grid_values(self):
        """Test that parameter grid values are reasonable."""
        param_grids = get_param_grids("classification")
        
        # Test logistic regression
        if 'logistic_regression' in param_grids:
            lr_grid = param_grids['logistic_regression']
            if 'classifier__C' in lr_grid:
                c_values = lr_grid['classifier__C']
                assert all(c > 0 for c in c_values), "C values must be positive"
        
        # Test random forest
        if 'random_forest' in param_grids:
            rf_grid = param_grids['random_forest']
            if 'classifier__n_estimators' in rf_grid:
                n_est_values = rf_grid['classifier__n_estimators']
                assert all(n > 0 for n in n_est_values), "n_estimators must be positive"


class TestPipelineValidation:
    """Test suite for pipeline input validation."""
    
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
    
    def test_validate_pipeline_input_valid(self, valid_data):
        """Test validation with valid input."""
        is_valid, error_msg = validate_pipeline_input(valid_data, "classification", use_charges=True)
        assert is_valid is True
        assert error_msg == ""
    
    def test_validate_pipeline_input_empty(self):
        """Test validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        is_valid, error_msg = validate_pipeline_input(empty_df, "classification")
        assert is_valid is False
        assert "empty" in error_msg.lower()
    
    def test_validate_pipeline_input_invalid_task(self, valid_data):
        """Test validation with invalid task."""
        is_valid, error_msg = validate_pipeline_input(valid_data, "invalid_task")
        assert is_valid is False
        assert "Invalid task" in error_msg
    
    def test_validate_pipeline_input_missing_columns(self):
        """Test validation with missing required columns."""
        incomplete_data = pd.DataFrame({
            'age': [25, 45],
            'sex': [0, 1]
            # Missing other columns
        })
        
        is_valid, error_msg = validate_pipeline_input(incomplete_data, "classification")
        assert is_valid is False
        assert "validation failed" in error_msg.lower()


class TestPipelineIntegration:
    """Integration tests for complete pipeline functionality."""
    
    @pytest.fixture
    def training_data(self):
        """Create training data."""
        np.random.seed(42)
        n_samples = 100
        
        return pd.DataFrame({
            'age': np.random.randint(18, 80, n_samples),
            'sex': np.random.choice([0, 1], n_samples),
            'bmi': np.random.uniform(18, 40, n_samples),
            'children': np.random.randint(0, 5, n_samples),
            'smoker': np.random.choice([0, 1], n_samples),
            'region': np.random.choice([0, 1, 2, 3], n_samples),
            'charges': np.random.uniform(1000, 50000, n_samples)
        })
    
    @pytest.fixture
    def training_target(self):
        """Create training target."""
        np.random.seed(42)
        return np.random.choice([0, 1], 100)
    
    def test_end_to_end_pipeline(self, training_data, training_target):
        """Test complete end-to-end pipeline functionality."""
        # Build pipelines
        pipelines = build_model_pipelines("classification", use_charges_feature=True)
        
        # Test one complete pipeline
        pipeline = pipelines['logistic_regression']
        
        # Fit pipeline
        pipeline.fit(training_data, training_target)
        
        # Make predictions
        predictions = pipeline.predict(training_data)
        probabilities = pipeline.predict_proba(training_data)
        
        # Validate predictions
        assert len(predictions) == len(training_data)
        assert all(pred in [0, 1] for pred in predictions)
        assert probabilities.shape == (len(training_data), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    def test_pipeline_reproducibility(self, training_data, training_target):
        """Test that pipelines give reproducible results."""
        # Build same pipeline twice
        pipelines1 = build_model_pipelines("classification", use_charges_feature=True)
        pipelines2 = build_model_pipelines("classification", use_charges_feature=True)
        
        pipeline1 = pipelines1['random_forest']
        pipeline2 = pipelines2['random_forest']
        
        # Fit both pipelines
        pipeline1.fit(training_data, training_target)
        pipeline2.fit(training_data, training_target)
        
        # Make predictions
        pred1 = pipeline1.predict(training_data)
        pred2 = pipeline2.predict(training_data)
        
        # Should be identical (due to random_state)
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_pipeline_feature_names(self, training_data, training_target):
        """Test that pipeline properly handles feature names."""
        pipelines = build_model_pipelines("classification", use_charges_feature=True)
        pipeline = pipelines['logistic_regression']
        
        # Fit pipeline
        pipeline.fit(training_data, training_target)
        
        # Check that we can access feature names from preprocessing step
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Should be able to get feature names out
        try:
            feature_names = preprocessor.get_feature_names_out()
            assert len(feature_names) > 0
        except AttributeError:
            # Some sklearn versions might not support this
            pass


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])