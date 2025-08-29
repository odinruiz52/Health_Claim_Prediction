#!/usr/bin/env python3
"""
🏥 Health Insurance Claim Prediction - Flask Web Application (FIXED)

CRITICAL FIXES IMPLEMENTED:
- ✅ Loads complete Pipeline (not separate model + scaler)
- ✅ Predicts on raw input data (no manual feature engineering)
- ✅ Uses centralized config for validation ranges and settings
- ✅ Proper error handling without information leakage
- ✅ Comprehensive logging with structured data

This Flask app now correctly:
1. Loads the complete trained Pipeline from model_development.py
2. Validates input using centralized config ranges
3. Passes raw data directly to Pipeline.predict()
4. Returns predictions with proper error handling

Author: Data Science Team  
Date: 2024
Repository: https://github.com/odinruiz52/Health_Claim_Prediction
"""

from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Add src to path for local imports
sys.path.append(str(Path(__file__).parent))

# Local imports
from config import get_config, setup_logging
from features import validate_feature_data

# Initialize configuration and logging
config = get_config()
setup_logging(config)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', config.flask.secret_key)

# Load the trained pipeline
MODEL_PATH = Path(config.model.model_path).resolve()
if not MODEL_PATH.exists():
    # Fallback to the new pipeline location
    MODEL_PATH = Path(__file__).parent.parent / "models" / "final_pipeline_classification.joblib"

try:
    if MODEL_PATH.exists():
        # Load the COMPLETE pipeline (not separate model + scaler)
        model_pipeline = joblib.load(MODEL_PATH)
        logger.info(f"✅ Loaded complete pipeline from: {MODEL_PATH}")
        
        # Verify it's a proper pipeline
        if not hasattr(model_pipeline, 'predict'):
            raise ValueError("Loaded object is not a valid predictor")
            
        MODEL_LOADED = True
    else:
        logger.error(f"❌ Model file not found: {MODEL_PATH}")
        model_pipeline = None
        MODEL_LOADED = False
        
except Exception as e:
    logger.error(f"❌ Error loading model: {str(e)}")
    model_pipeline = None
    MODEL_LOADED = False


def validate_input_payload(payload: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Validate input payload using centralized config validation.
    
    Parameters:
    -----------
    payload : dict
        Input data from request
        
    Returns:
    --------
    tuple[bool, str, dict | None]
        (is_valid, error_message, cleaned_data)
    """
    try:
        # Extract and convert data types
        cleaned_data = {
            'age': int(payload.get('age', 0)),
            'sex': int(payload.get('sex', -1)),
            'bmi': float(payload.get('bmi', 0.0)),
            'children': int(payload.get('children', 0)), 
            'smoker': int(payload.get('smoker', -1)),
            'region': int(payload.get('region', -1)),
        }
        
        # Add charges if provided (optional for inference)
        if 'charges' in payload:
            cleaned_data['charges'] = float(payload['charges'])
            
    except (ValueError, TypeError) as e:
        return False, "Invalid data types in input", None
    
    # Validate using centralized ranges from config
    try:
        # Age validation
        if not (config.model.age_min <= cleaned_data['age'] <= config.model.age_max):
            return False, f"Age must be between {config.model.age_min} and {config.model.age_max}", None
            
        # BMI validation  
        if not (config.model.bmi_min <= cleaned_data['bmi'] <= config.model.bmi_max):
            return False, f"BMI must be between {config.model.bmi_min} and {config.model.bmi_max}", None
            
        # Children validation
        if not (config.model.children_min <= cleaned_data['children'] <= config.model.children_max):
            return False, f"Children must be between {config.model.children_min} and {config.model.children_max}", None
            
        # Categorical validations
        if cleaned_data['sex'] not in [0, 1]:
            return False, "Sex must be 0 (female) or 1 (male)", None
            
        if cleaned_data['smoker'] not in [0, 1]:
            return False, "Smoker must be 0 (no) or 1 (yes)", None
            
        if cleaned_data['region'] not in [0, 1, 2, 3]:
            return False, "Region must be 0, 1, 2, or 3", None
            
        # Charges validation (if provided)
        if 'charges' in cleaned_data:
            if not (config.model.charges_min <= cleaned_data['charges'] <= config.model.charges_max):
                return False, f"Charges must be between {config.model.charges_min} and {config.model.charges_max}", None
                
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return False, "Validation configuration error", None
    
    # Additional validation using feature validation
    df_temp = pd.DataFrame([cleaned_data])
    is_valid, error_msg = validate_feature_data(df_temp, require_charges='charges' in cleaned_data)
    if not is_valid:
        return False, f"Data validation failed: {error_msg}", None
        
    logger.debug(f"Input validation passed for payload: {cleaned_data}")
    return True, "", cleaned_data


@app.route('/')
def home():
    """Serve the main page."""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving home page: {str(e)}")
        return jsonify({"error": "Application error"}), 500


@app.route('/health')
def health_check():
    """Health check endpoint."""
    health_status = {
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED,
        "model_path": str(MODEL_PATH),
        "version": "2.0.0"
    }
    
    status_code = 200 if MODEL_LOADED else 503
    return jsonify(health_status), status_code


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions using the trained pipeline.
    
    Expected input (JSON or form data):
    {
        "age": 35,
        "sex": 1,
        "bmi": 26.5, 
        "children": 2,
        "smoker": 0,
        "region": 1,
        "charges": 5000.0  # Optional
    }
    
    Returns:
    {
        "prediction": "Approved" or "Denied",
        "confidence": 0.85,
        "status": "success"
    }
    """
    # Check model availability
    if not MODEL_LOADED or model_pipeline is None:
        logger.error("Prediction attempted but model not loaded")
        return jsonify({
            "error": "Model not available",
            "status": "error"
        }), 503
    
    try:
        # Parse input data
        if request.is_json:
            payload = request.get_json()
        else:
            payload = request.form.to_dict()
            
        if not payload:
            return jsonify({
                "error": "No input data provided", 
                "status": "error"
            }), 400
        
        # Validate input
        is_valid, error_msg, cleaned_data = validate_input_payload(payload)
        if not is_valid:
            logger.warning(f"Invalid input: {error_msg}")
            return jsonify({
                "error": error_msg,
                "status": "error"
            }), 400
        
        # Create DataFrame with raw input (Pipeline handles all preprocessing)
        input_df = pd.DataFrame([cleaned_data])
        
        # Make prediction using the complete pipeline
        prediction = model_pipeline.predict(input_df)
        prediction_value = int(prediction[0])
        
        # Get prediction probabilities if available
        confidence = None
        if hasattr(model_pipeline, 'predict_proba'):
            try:
                proba = model_pipeline.predict_proba(input_df)
                confidence = float(max(proba[0]))
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {str(e)}")
                confidence = None
        
        # Convert prediction to human-readable format
        # Assuming 0 = Approved, 1 = Denied based on typical healthcare claim encoding
        prediction_label = "Denied" if prediction_value == 1 else "Approved"
        
        # Log successful prediction (without sensitive data)
        logger.info(
            "Prediction completed successfully",
            extra={
                "prediction": prediction_label,
                "confidence": confidence,
                "model_path": str(MODEL_PATH)
            }
        )
        
        # Prepare response
        response_data = {
            "prediction": prediction_label,
            "status": "success"
        }
        
        if confidence is not None:
            response_data["confidence"] = round(confidence, 3)
            
        # Handle different response formats
        if request.is_json:
            return jsonify(response_data), 200
        else:
            # For form submissions, render template with results
            prediction_text = f"Predicted Claim Status: {prediction_label}"
            if confidence is not None:
                prediction_text += f" (Confidence: {confidence:.1%})"
                
            error_class = 'success' if prediction_label == 'Approved' else 'warning'
            
            return render_template(
                'index.html',
                prediction_text=prediction_text,
                error_class=error_class
            )
    
    except Exception as e:
        # Log error without exposing internal details
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        
        error_response = {
            "error": "Prediction failed - please check your input and try again",
            "status": "error"
        }
        
        if request.is_json:
            return jsonify(error_response), 500
        else:
            return render_template(
                'index.html',
                prediction_text="An error occurred during prediction. Please try again.",
                error_class='error'
            )


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found", "status": "error"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error", "status": "error"}), 500


@app.before_first_request
def startup_check():
    """Perform startup checks."""
    logger.info("🏥 Health Insurance Prediction App Starting")
    logger.info(f"Model loaded: {MODEL_LOADED}")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Config environment: {config.environment}")
    
    if not MODEL_LOADED:
        logger.warning("⚠️ App started without a loaded model - predictions will fail")


def create_app(config_name: str = None) -> Flask:
    """
    Application factory pattern.
    
    Parameters:
    -----------
    config_name : str, optional
        Configuration environment name
        
    Returns:
    --------
    Flask
        Configured Flask application
    """
    return app


if __name__ == '__main__':
    # Development server
    logger.info("Starting Flask development server...")
    
    # Check if model is available
    if not MODEL_LOADED:
        logger.warning(
            "⚠️ No model loaded! Train a model first by running: python model_development.py"
        )
    
    app.run(
        host=config.flask.host,
        port=config.flask.port, 
        debug=config.flask.debug,
        threaded=True
    )