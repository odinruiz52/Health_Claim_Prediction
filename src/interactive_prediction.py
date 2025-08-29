#!/usr/bin/env python3
"""
🏥 Health Insurance Claim Prediction - Interactive CLI Tool (FIXED)

CRITICAL FIXES IMPLEMENTED:
- ✅ Loads complete Pipeline (not separate model + scaler)
- ✅ Uses raw input data (no manual feature engineering)
- ✅ Shares validation logic with Flask app (consistency)
- ✅ Proper error handling and user feedback
- ✅ Clean separation of concerns

This CLI tool now correctly:
1. Loads the same Pipeline as the web app
2. Uses centralized validation
3. Passes raw data directly to Pipeline.predict()  
4. Provides interactive user experience

Author: Data Science Team
Date: 2024
Repository: https://github.com/odinruiz52/Health_Claim_Prediction
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import pandas as pd

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Local imports
from config import get_config, setup_logging
from features import validate_feature_data

# Initialize configuration and logging
config = get_config()
setup_logging(config)
logger = logging.getLogger(__name__)

# Global variables for model
model_pipeline = None
MODEL_LOADED = False


def load_model_pipeline() -> bool:
    """
    Load the trained pipeline.
    
    Returns:
    --------
    bool
        True if model loaded successfully, False otherwise
    """
    global model_pipeline, MODEL_LOADED
    
    # Try primary model path from config
    model_path = Path(config.model.model_path).resolve()
    
    # Fallback to new pipeline location
    if not model_path.exists():
        model_path = Path(__file__).parent.parent / "models" / "final_pipeline_classification.joblib"
    
    try:
        if model_path.exists():
            model_pipeline = joblib.load(model_path)
            
            # Verify it's a valid pipeline
            if not hasattr(model_pipeline, 'predict'):
                raise ValueError("Loaded object is not a valid predictor")
                
            logger.info(f"✅ Loaded pipeline: {model_path}")
            MODEL_LOADED = True
            return True
        else:
            logger.error(f"❌ Model file not found: {model_path}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return False


def validate_user_input(user_data: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate user input using centralized validation logic.
    
    Parameters:
    -----------
    user_data : dict
        User input data
        
    Returns:
    --------
    tuple[bool, str]
        (is_valid, error_message)
    """
    try:
        # Validate using centralized config ranges
        if not (config.model.age_min <= user_data['age'] <= config.model.age_max):
            return False, f"Age must be between {config.model.age_min} and {config.model.age_max}"
            
        if not (config.model.bmi_min <= user_data['bmi'] <= config.model.bmi_max):
            return False, f"BMI must be between {config.model.bmi_min} and {config.model.bmi_max}"
            
        if not (config.model.children_min <= user_data['children'] <= config.model.children_max):
            return False, f"Children must be between {config.model.children_min} and {config.model.children_max}"
            
        if user_data['sex'] not in [0, 1]:
            return False, "Sex must be 0 (female) or 1 (male)"
            
        if user_data['smoker'] not in [0, 1]:
            return False, "Smoker must be 0 (no) or 1 (yes)"
            
        if user_data['region'] not in [0, 1, 2, 3]:
            return False, "Region must be 0, 1, 2, or 3"
            
        # Optional charges validation
        if 'charges' in user_data:
            if not (config.model.charges_min <= user_data['charges'] <= config.model.charges_max):
                return False, f"Charges must be between {config.model.charges_min} and {config.model.charges_max}"
        
        # Additional validation using feature validation function
        df_temp = pd.DataFrame([user_data])
        is_valid, error_msg = validate_feature_data(df_temp, require_charges='charges' in user_data)
        if not is_valid:
            return False, error_msg
            
        return True, ""
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return False, f"Validation error: {str(e)}"


def make_prediction(user_data: Dict[str, Any]) -> tuple[str, Optional[float]]:
    """
    Make prediction using the loaded pipeline.
    
    Parameters:
    -----------
    user_data : dict
        User input data
        
    Returns:
    --------
    tuple[str, float | None]
        (prediction_label, confidence)
    """
    if not MODEL_LOADED or model_pipeline is None:
        raise RuntimeError("Model not loaded")
    
    # Create DataFrame from raw input (Pipeline handles all preprocessing)
    input_df = pd.DataFrame([user_data])
    
    # Make prediction
    prediction = model_pipeline.predict(input_df)
    prediction_value = int(prediction[0])
    
    # Get confidence if available
    confidence = None
    if hasattr(model_pipeline, 'predict_proba'):
        try:
            proba = model_pipeline.predict_proba(input_df)
            confidence = float(max(proba[0]))
        except Exception as e:
            logger.warning(f"Could not get prediction probabilities: {str(e)}")
    
    # Convert to human-readable format
    prediction_label = "Denied" if prediction_value == 1 else "Approved"
    
    return prediction_label, confidence


def get_user_input_interactive() -> Optional[Dict[str, Any]]:
    """
    Get user input interactively with validation.
    
    Returns:
    --------
    dict | None
        User input data, or None if user wants to quit
    """
    print("\n📝 Enter patient information:")
    print("=" * 40)
    
    try:
        # Get basic information
        age = int(input(f"Age ({config.model.age_min}-{config.model.age_max}): ").strip())
        sex = int(input("Sex (0=Female, 1=Male): ").strip())
        bmi = float(input(f"BMI ({config.model.bmi_min}-{config.model.bmi_max}): ").strip())
        children = int(input(f"Number of children ({config.model.children_min}-{config.model.children_max}): ").strip())
        smoker = int(input("Smoker (0=No, 1=Yes): ").strip())
        region = int(input("Region (0=Northeast, 1=Northwest, 2=Southeast, 3=Southwest): ").strip())
        
        user_data = {
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region
        }
        
        # Optional: Ask for charges if available
        charges_input = input(f"Medical charges ({config.model.charges_min}-{config.model.charges_max}, press Enter to skip): ").strip()
        if charges_input:
            user_data['charges'] = float(charges_input)
            
        return user_data
        
    except ValueError as e:
        print(f"❌ Invalid input: Please enter valid numbers")
        return None
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        return None


def predict_from_dataset() -> None:
    """Predict using a record from the dataset."""
    
    # Load dataset
    data_path = Path(__file__).parent.parent / "data" / "insurance_encoded.csv"
    
    try:
        df = pd.read_csv(data_path)
        
        if 'claim_status' in df.columns:
            feature_df = df.drop(columns=['claim_status'])
            target_series = df['claim_status']
        else:
            feature_df = df
            target_series = None
            
        print(f"\n📊 Dataset contains {len(df)} records (indices 0 to {len(df)-1})")
        
        try:
            index = int(input("Enter row index: ").strip())
            
            if not (0 <= index < len(df)):
                print(f"❌ Index must be between 0 and {len(df)-1}")
                return
                
            # Get the record
            record_data = feature_df.iloc[index].to_dict()
            
            print(f"\n📋 Selected record #{index}:")
            for key, value in record_data.items():
                print(f"  {key}: {value}")
                
            # Show actual status if available
            if target_series is not None:
                actual_status = "Denied" if target_series.iloc[index] == 1 else "Approved"
                print(f"\n🎯 Actual Status: {actual_status}")
            
            # Make prediction
            predicted_status, confidence = make_prediction(record_data)
            
            print(f"\n🤖 Predicted Status: {predicted_status}")
            if confidence is not None:
                print(f"📊 Confidence: {confidence:.1%}")
                
            # Compare if actual available
            if target_series is not None:
                if predicted_status == actual_status:
                    print("✅ Prediction matches actual status!")
                else:
                    print("❌ Prediction differs from actual status")
            
        except ValueError:
            print("❌ Please enter a valid number")
            return
            
    except FileNotFoundError:
        print(f"❌ Dataset not found: {data_path}")
        print("Please ensure the data file exists before using this option.")
        return
    except Exception as e:
        logger.error(f"Error reading dataset: {str(e)}")
        print("❌ Error reading dataset")
        return


def show_dataset_statistics() -> None:
    """Display dataset statistics."""
    
    data_path = Path(__file__).parent.parent / "data" / "insurance_encoded.csv"
    
    try:
        df = pd.read_csv(data_path)
        
        print(f"\n📈 Dataset Statistics:")
        print(f"Total records: {len(df)}")
        
        if 'claim_status' in df.columns:
            approved = (df['claim_status'] == 0).sum()
            denied = (df['claim_status'] == 1).sum()
            print(f"Approved claims: {approved} ({approved/len(df)*100:.1f}%)")
            print(f"Denied claims: {denied} ({denied/len(df)*100:.1f}%)")
        
        print("\nFeature ranges:")
        feature_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        if 'charges' in df.columns:
            feature_cols.append('charges')
            
        for col in feature_cols:
            if col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    print(f"  {col}: {df[col].min():.1f} - {df[col].max():.1f}")
                else:
                    unique_vals = df[col].unique()
                    print(f"  {col}: {list(unique_vals)}")
                    
    except Exception as e:
        logger.error(f"Error reading dataset statistics: {str(e)}")
        print("❌ Error reading dataset statistics")


def main_menu() -> None:
    """Display main menu and handle user interaction."""
    
    print("\n🏥 Health Insurance Claim Prediction Tool")
    print("=" * 50)
    print("Select an option:")
    print("1. Predict using custom patient data")
    print("2. Predict using dataset record")
    print("3. Show dataset statistics")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                # Custom prediction
                user_data = get_user_input_interactive()
                if user_data is None:
                    continue
                    
                # Validate input
                is_valid, error_msg = validate_user_input(user_data)
                if not is_valid:
                    print(f"❌ Input Error: {error_msg}")
                    continue
                    
                # Make prediction
                try:
                    prediction, confidence = make_prediction(user_data)
                    
                    print(f"\n🤖 Predicted Claim Status: {prediction}")
                    if confidence is not None:
                        print(f"📊 Confidence: {confidence:.1%}")
                        
                    if prediction == 'Approved':
                        print("✅ This claim is likely to be approved")
                    else:
                        print("⚠️ This claim is likely to be denied")
                        
                except Exception as e:
                    logger.error(f"Prediction error: {str(e)}")
                    print("❌ Error making prediction")
                    
            elif choice == '2':
                # Dataset prediction
                predict_from_dataset()
                
            elif choice == '3':
                # Dataset statistics
                show_dataset_statistics()
                
            elif choice == '4':
                print("👋 Thank you for using the Health Insurance Claim Prediction Tool!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            print("❌ An unexpected error occurred")


def main():
    """Main entry point."""
    
    print("🏥 Health Insurance Claim Prediction - Interactive CLI")
    print("Repository: https://github.com/odinruiz52/Health_Claim_Prediction")
    print("=" * 70)
    
    # Load model
    print("Loading model pipeline...")
    if not load_model_pipeline():
        print("❌ Failed to load model!")
        print("\n💡 To train a model, run: python model_development.py")
        print("Then try running this tool again.")
        return
        
    print("✅ Model loaded successfully!")
    
    try:
        main_menu()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print("❌ Application error occurred")


if __name__ == "__main__":
    main()