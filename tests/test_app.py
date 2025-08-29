#!/usr/bin/env python3
"""
🏥 Health Insurance Claim Prediction - Flask App Tests

Tests for the Flask web application to ensure:
- Proper request/response handling
- Input validation consistency
- Error handling without information leakage
- Integration with the ML pipeline

Author: Data Science Team
Date: 2024
Repository: https://github.com/odinruiz52/Health_Claim_Prediction
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import the Flask app
from app import app, validate_input_payload


class TestFlaskApp:
    """Test suite for Flask application."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    @pytest.fixture
    def valid_payload(self):
        """Valid input payload for testing."""
        return {
            'age': '35',
            'sex': '1',
            'bmi': '26.5',
            'children': '2',
            'smoker': '0',
            'region': '1',
            'charges': '5000.0'
        }
    
    def test_home_route(self, client):
        """Test the home route."""
        response = client.get('/')
        assert response.status_code == 200
        # Should render template (will fail if template missing, that's expected for now)
    
    def test_health_check_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get('/health')
        assert response.status_code in [200, 503]  # Depends on model availability
        
        data = json.loads(response.data)
        assert 'status' in data
        assert 'model_loaded' in data
        assert 'version' in data
        assert data['status'] in ['healthy', 'unhealthy']
        assert isinstance(data['model_loaded'], bool)
    
    @patch('src.app.MODEL_LOADED', False)
    @patch('src.app.model_pipeline', None)
    def test_predict_no_model(self, client, valid_payload):
        """Test prediction when model is not loaded."""
        response = client.post('/predict', 
                              data=json.dumps(valid_payload),
                              content_type='application/json')
        
        assert response.status_code == 503
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'Model not available' in data['error']
    
    def test_predict_no_data(self, client):
        """Test prediction with no input data."""
        response = client.post('/predict',
                              data=json.dumps({}),
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'No input data provided' in data['error']
    
    def test_predict_invalid_json(self, client):
        """Test prediction with invalid JSON."""
        response = client.post('/predict',
                              data="invalid json",
                              content_type='application/json')
        
        assert response.status_code == 400
    
    @patch('src.app.MODEL_LOADED', True)
    def test_predict_invalid_input_validation(self, client):
        """Test prediction with invalid input that fails validation."""
        invalid_payload = {
            'age': '200',  # Out of range
            'sex': '1',
            'bmi': '26.5',
            'children': '2',
            'smoker': '0',
            'region': '1'
        }
        
        response = client.post('/predict',
                              data=json.dumps(invalid_payload),
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'Age must be between' in data['error']
    
    @patch('src.app.MODEL_LOADED', True)
    @patch('src.app.model_pipeline')
    def test_predict_success_json(self, mock_pipeline, client, valid_payload):
        """Test successful prediction with JSON input."""
        # Mock the pipeline
        mock_pipeline.predict.return_value = [0]  # Approved
        mock_pipeline.predict_proba.return_value = [[0.3, 0.7]]
        
        response = client.post('/predict',
                              data=json.dumps(valid_payload),
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['prediction'] in ['Approved', 'Denied']
        assert 'confidence' in data
        assert 0 <= data['confidence'] <= 1
    
    @patch('src.app.MODEL_LOADED', True)
    @patch('src.app.model_pipeline')
    def test_predict_success_form_data(self, mock_pipeline, client, valid_payload):
        """Test successful prediction with form data."""
        # Mock the pipeline
        mock_pipeline.predict.return_value = [1]  # Denied
        mock_pipeline.predict_proba.return_value = [[0.2, 0.8]]
        
        response = client.post('/predict', data=valid_payload)
        
        assert response.status_code == 200
        # Form data returns HTML template
        assert response.content_type.startswith('text/html')
    
    @patch('src.app.MODEL_LOADED', True)  
    @patch('src.app.model_pipeline')
    def test_predict_pipeline_error(self, mock_pipeline, client, valid_payload):
        """Test prediction when pipeline throws error."""
        # Mock pipeline to raise exception
        mock_pipeline.predict.side_effect = Exception("Pipeline error")
        
        response = client.post('/predict',
                              data=json.dumps(valid_payload),
                              content_type='application/json')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'Prediction failed' in data['error']
        # Should not expose internal error details
        assert 'Pipeline error' not in data['error']
    
    def test_404_error_handler(self, client):
        """Test 404 error handler."""
        response = client.get('/nonexistent')
        assert response.status_code == 404
        
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'not found' in data['error'].lower()


class TestInputValidation:
    """Test suite for input validation function."""
    
    def test_validate_input_payload_valid(self):
        """Test validation with valid payload."""
        valid_payload = {
            'age': '35',
            'sex': '1', 
            'bmi': '26.5',
            'children': '2',
            'smoker': '0',
            'region': '1',
            'charges': '5000.0'
        }
        
        is_valid, error_msg, cleaned_data = validate_input_payload(valid_payload)
        
        assert is_valid is True
        assert error_msg == ""
        assert cleaned_data is not None
        assert cleaned_data['age'] == 35
        assert cleaned_data['bmi'] == 26.5
        assert cleaned_data['charges'] == 5000.0
    
    def test_validate_input_payload_invalid_types(self):
        """Test validation with invalid data types."""
        invalid_payload = {
            'age': 'not_a_number',
            'sex': '1',
            'bmi': '26.5',
            'children': '2',
            'smoker': '0',
            'region': '1'
        }
        
        is_valid, error_msg, cleaned_data = validate_input_payload(invalid_payload)
        
        assert is_valid is False
        assert 'Invalid data types' in error_msg
        assert cleaned_data is None
    
    def test_validate_input_payload_age_range(self):
        """Test validation with age out of range."""
        invalid_payload = {
            'age': '200',  # Out of range
            'sex': '1',
            'bmi': '26.5',
            'children': '2',
            'smoker': '0',
            'region': '1'
        }
        
        is_valid, error_msg, cleaned_data = validate_input_payload(invalid_payload)
        
        assert is_valid is False
        assert 'Age must be between' in error_msg
        assert cleaned_data is None
    
    def test_validate_input_payload_bmi_range(self):
        """Test validation with BMI out of range."""
        invalid_payload = {
            'age': '35',
            'sex': '1',
            'bmi': '80.0',  # Out of range
            'children': '2',
            'smoker': '0',
            'region': '1'
        }
        
        is_valid, error_msg, cleaned_data = validate_input_payload(invalid_payload)
        
        assert is_valid is False
        assert 'BMI must be between' in error_msg
        assert cleaned_data is None
    
    def test_validate_input_payload_categorical_values(self):
        """Test validation with invalid categorical values."""
        invalid_payloads = [
            {'age': '35', 'sex': '9', 'bmi': '26.5', 'children': '2', 'smoker': '0', 'region': '1'},  # Invalid sex
            {'age': '35', 'sex': '1', 'bmi': '26.5', 'children': '2', 'smoker': '9', 'region': '1'},  # Invalid smoker
            {'age': '35', 'sex': '1', 'bmi': '26.5', 'children': '2', 'smoker': '0', 'region': '9'},  # Invalid region
        ]
        
        for payload in invalid_payloads:
            is_valid, error_msg, cleaned_data = validate_input_payload(payload)
            assert is_valid is False
            assert cleaned_data is None
            # Should contain specific error message
            assert any(word in error_msg for word in ['Sex', 'Smoker', 'Region'])
    
    def test_validate_input_payload_missing_fields(self):
        """Test validation with missing required fields."""
        incomplete_payload = {
            'age': '35',
            'sex': '1'
            # Missing other required fields
        }
        
        is_valid, error_msg, cleaned_data = validate_input_payload(incomplete_payload)
        
        assert is_valid is False
        assert cleaned_data is None
    
    def test_validate_input_payload_optional_charges(self):
        """Test validation with and without optional charges field."""
        # Without charges
        payload_no_charges = {
            'age': '35',
            'sex': '1',
            'bmi': '26.5',
            'children': '2',
            'smoker': '0',
            'region': '1'
        }
        
        is_valid, error_msg, cleaned_data = validate_input_payload(payload_no_charges)
        assert is_valid is True
        assert 'charges' not in cleaned_data
        
        # With charges
        payload_with_charges = payload_no_charges.copy()
        payload_with_charges['charges'] = '5000.0'
        
        is_valid, error_msg, cleaned_data = validate_input_payload(payload_with_charges)
        assert is_valid is True
        assert 'charges' in cleaned_data
        assert cleaned_data['charges'] == 5000.0
    
    def test_validate_input_payload_charges_range(self):
        """Test validation of charges range."""
        invalid_payload = {
            'age': '35',
            'sex': '1',
            'bmi': '26.5',
            'children': '2', 
            'smoker': '0',
            'region': '1',
            'charges': '1000000.0'  # Out of range
        }
        
        is_valid, error_msg, cleaned_data = validate_input_payload(invalid_payload)
        
        assert is_valid is False
        assert 'Charges must be between' in error_msg
        assert cleaned_data is None


class TestAppIntegration:
    """Integration tests for the Flask application."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_app_configuration(self):
        """Test app configuration."""
        assert app.config['TESTING'] is True
        assert app.secret_key is not None
    
    def test_cors_headers(self, client):
        """Test that appropriate headers are set."""
        response = client.get('/health')
        # Should not have CORS issues in testing
        assert response.status_code in [200, 503]
    
    @patch('src.app.MODEL_LOADED', True)
    @patch('src.app.model_pipeline')  
    def test_prediction_data_flow(self, mock_pipeline, client):
        """Test the complete data flow for prediction."""
        # Mock pipeline behavior
        mock_pipeline.predict.return_value = [0]
        mock_pipeline.predict_proba.return_value = [[0.4, 0.6]]
        
        # Test input data
        test_payload = {
            'age': '45',
            'sex': '0',
            'bmi': '28.5',
            'children': '3',
            'smoker': '1',
            'region': '2'
        }
        
        # Send request
        response = client.post('/predict',
                              data=json.dumps(test_payload),
                              content_type='application/json')
        
        # Check response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['prediction'] == 'Approved'  # 0 = Approved
        
        # Verify pipeline was called with correct data
        mock_pipeline.predict.assert_called_once()
        call_args = mock_pipeline.predict.call_args[0][0]  # First positional argument
        assert isinstance(call_args, pd.DataFrame)
        assert len(call_args) == 1  # Single row
        assert call_args.iloc[0]['age'] == 45
        assert call_args.iloc[0]['sex'] == 0
        assert call_args.iloc[0]['smoker'] == 1


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])