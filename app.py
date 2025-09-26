#!/usr/bin/env python3
"""
Simple Flask web application for health insurance claim prediction.
"""

import sys
if sys.version_info[:2] != (3, 11):
    print(f"[INFO] This project was tested on Python 3.11.x "
          f"(you're on {sys.version_info.major}.{sys.version_info.minor}). "
          "If you hit install/build issues, please switch to Python 3.11.")

import os
import joblib
import pandas as pd
from flask import Flask, request, render_template_string, jsonify
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev_secret_key_change_in_production')

# Load model
MODEL_PATH = os.getenv('MODEL_PATH', 'models/logistic_regression.joblib')
model = None

def load_model():
    """Load the trained model."""
    global model
    try:
        if Path(MODEL_PATH).exists():
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded: {MODEL_PATH}")
            return True
        else:
            print(f"Model not found: {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Health Insurance Claim Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin: 20px 0; padding: 15px; border-radius: 4px; }
        .success { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .error { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .warning { background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }
    </style>
</head>
<body>
    <h1>Health Insurance Claim Prediction</h1>

    {% if prediction %}
    <div class="result {{ result_class }}">
        <strong>Prediction Result:</strong> {{ prediction }}
    </div>
    {% endif %}

    {% if error %}
    <div class="result error">
        <strong>Error:</strong> {{ error }}
    </div>
    {% endif %}

    <form method="POST" action="/predict">
        <div class="form-group">
            <label for="age">Age (18-100):</label>
            <input type="number" id="age" name="age" min="18" max="100" required>
        </div>

        <div class="form-group">
            <label for="sex">Sex:</label>
            <select id="sex" name="sex" required>
                <option value="">Select...</option>
                <option value="0">Female</option>
                <option value="1">Male</option>
            </select>
        </div>

        <div class="form-group">
            <label for="bmi">BMI (15.0-50.0):</label>
            <input type="number" id="bmi" name="bmi" min="15" max="50" step="0.1" required>
        </div>

        <div class="form-group">
            <label for="children">Number of Children (0-10):</label>
            <input type="number" id="children" name="children" min="0" max="10" required>
        </div>

        <div class="form-group">
            <label for="smoker">Smoker:</label>
            <select id="smoker" name="smoker" required>
                <option value="">Select...</option>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select>
        </div>

        <div class="form-group">
            <label for="region">Region:</label>
            <select id="region" name="region" required>
                <option value="">Select...</option>
                <option value="0">Northeast</option>
                <option value="1">Southeast</option>
                <option value="2">Southwest</option>
                <option value="3">Northwest</option>
            </select>
        </div>

        <button type="submit">Predict Claim Status</button>
    </form>
</body>
</html>
"""

@app.route('/')
def home():
    """Home page with prediction form."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on form input."""
    if model is None:
        return render_template_string(
            HTML_TEMPLATE,
            error="Model not loaded. Please train a model first.",
            result_class="error"
        )

    try:
        # Get form data
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        region = int(request.form['region'])

        # Validate inputs
        if not (18 <= age <= 100):
            raise ValueError("Age must be between 18 and 100")
        if sex not in [0, 1]:
            raise ValueError("Sex must be 0 (Female) or 1 (Male)")
        if not (15.0 <= bmi <= 50.0):
            raise ValueError("BMI must be between 15.0 and 50.0")
        if not (0 <= children <= 10):
            raise ValueError("Children must be between 0 and 10")
        if smoker not in [0, 1]:
            raise ValueError("Smoker must be 0 (No) or 1 (Yes)")
        if region not in [0, 1, 2, 3]:
            raise ValueError("Region must be 0, 1, 2, or 3")

        # Create input DataFrame
        input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                                  columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Convert prediction to readable format
        result = "Approved" if prediction == 1 else "Denied"
        result_class = "success" if prediction == 1 else "error"

        # Return result
        if request.is_json:
            return jsonify({
                "prediction": result,
                "status": "success"
            })
        else:
            return render_template_string(
                HTML_TEMPLATE,
                prediction=result,
                result_class=result_class
            )

    except ValueError as e:
        error_msg = str(e)
        if request.is_json:
            return jsonify({"error": error_msg, "status": "error"}), 400
        else:
            return render_template_string(
                HTML_TEMPLATE,
                error=error_msg,
                result_class="error"
            )
    except Exception as e:
        error_msg = "Prediction failed. Please check your input."
        if request.is_json:
            return jsonify({"error": error_msg, "status": "error"}), 500
        else:
            return render_template_string(
                HTML_TEMPLATE,
                error=error_msg,
                result_class="error"
            )

if __name__ == '__main__':
    # Load model on startup
    model_loaded = load_model()

    if not model_loaded:
        print("Warning: No model loaded. Train a model first by running: python run_pipeline.py")

    # Get port from environment variable
    port = int(os.getenv('FLASK_PORT', 5000))

    # Run Flask app
    print(f"Starting Flask app on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)