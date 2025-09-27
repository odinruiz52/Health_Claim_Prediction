#!/usr/bin/env python3
"""
Simple Flask web application for health insurance claim prediction.
"""

import os
import joblib
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
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

def _clean_form(form_data):
    """Clean and convert form data to appropriate types."""
    try:
        return {
            'age': int(form_data.get('age', 0)),
            'sex': int(form_data.get('sex', 0)),
            'bmi': float(form_data.get('bmi', 0.0)),
            'children': int(form_data.get('children', 0)),
            'smoker': int(form_data.get('smoker', 0)),
            'region': int(form_data.get('region', 0))
        }
    except (ValueError, TypeError):
        return None

@app.route('/')
def home():
    """Home page with prediction form."""
    form_values = session.get('last_input', {})
    history = session.get('history', [])
    return render_template('index.html',
                         form_values=form_values,
                         history=history,
                         result=None)

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on form input."""
    if model is None:
        if request.is_json:
            return jsonify({"error": "Model not loaded", "status": "error"}), 500
        form_values = session.get('last_input', {})
        history = session.get('history', [])
        return render_template('index.html',
                             form_values=form_values,
                             history=history,
                             result=None,
                             error="Model not loaded. Please train a model first.")

    # Handle JSON API requests
    if request.is_json:
        try:
            data = request.get_json()
            age = int(data['age'])
            sex = int(data['sex'])
            bmi = float(data['bmi'])
            children = int(data['children'])
            smoker = int(data['smoker'])
            region = int(data['region'])

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
            decision = "Approved" if prediction == 1 else "Denied"

            return jsonify({
                "decision": decision,
                "status": "success"
            })

        except (KeyError, ValueError, TypeError) as e:
            return jsonify({"error": str(e), "status": "error"}), 400
        except Exception as e:
            return jsonify({"error": "Prediction failed", "status": "error"}), 500

    # Handle HTML form requests
    try:
        # Clean form data
        inputs = _clean_form(request.form)
        if inputs is None:
            raise ValueError("Invalid input data")

        # Validate inputs
        if not (18 <= inputs['age'] <= 100):
            raise ValueError("Age must be between 18 and 100")
        if inputs['sex'] not in [0, 1]:
            raise ValueError("Sex must be 0 (Female) or 1 (Male)")
        if not (15.0 <= inputs['bmi'] <= 50.0):
            raise ValueError("BMI must be between 15.0 and 50.0")
        if not (0 <= inputs['children'] <= 10):
            raise ValueError("Children must be between 0 and 10")
        if inputs['smoker'] not in [0, 1]:
            raise ValueError("Smoker must be 0 (No) or 1 (Yes)")
        if inputs['region'] not in [0, 1, 2, 3]:
            raise ValueError("Region must be 0, 1, 2, or 3")

        # Store form values in session for sticky inputs
        session['last_input'] = inputs

        # Create input DataFrame
        input_data = pd.DataFrame([[inputs['age'], inputs['sex'], inputs['bmi'],
                                   inputs['children'], inputs['smoker'], inputs['region']]],
                                  columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

        # Make prediction
        prediction = model.predict(input_data)[0]
        decision = "Approved" if prediction == 1 else "Denied"

        result = {"decision": decision}

        # Add to history (cap at 10)
        if 'history' not in session:
            session['history'] = []

        history_entry = {
            **inputs,
            'decision': decision,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        session['history'].insert(0, history_entry)  # Add to beginning
        if len(session['history']) > 10:
            session['history'] = session['history'][:10]  # Keep only 10

        # Get updated history for rendering
        history = session.get('history', [])

        return render_template('index.html',
                             form_values=inputs,
                             history=history,
                             result=result)

    except ValueError as e:
        form_values = session.get('last_input', {})
        history = session.get('history', [])
        return render_template('index.html',
                             form_values=form_values,
                             history=history,
                             result=None,
                             error=str(e))
    except Exception as e:
        form_values = session.get('last_input', {})
        history = session.get('history', [])
        return render_template('index.html',
                             form_values=form_values,
                             history=history,
                             result=None,
                             error="Prediction failed. Please check your input.")

@app.route('/clear', methods=['POST'])
def clear_form():
    """Clear form fields only."""
    if 'last_input' in session:
        del session['last_input']
    return redirect(url_for('home'))

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear prediction history only."""
    if 'history' in session:
        del session['history']
    return redirect(url_for('home'))

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