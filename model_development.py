#!/usr/bin/env python3
"""
🏥 Health Insurance Claim Prediction - Model Development Pipeline (FIXED)

CRITICAL FIXES IMPLEMENTED:
- ✅ Eliminates CV leakage by using complete Pipelines in GridSearchCV  
- ✅ Proper categorical variable handling with OneHotEncoder
- ✅ Unified preprocessing for all algorithms
- ✅ Saves complete Pipeline (not separate model + scaler)
- ✅ Consistent feature engineering between train and inference

This pipeline implements the corrected ML workflow:
1. Load data and define target
2. Create complete sklearn Pipelines (FeatureBuilder + Preprocessor + Estimator)
3. Perform GridSearchCV over entire Pipelines (no leakage)
4. Evaluate best Pipeline on holdout test set
5. Save complete Pipeline for inference

Author: Data Science Team
Date: 2024
Repository: https://github.com/odinruiz52/Health_Claim_Prediction
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV
)

# Local imports - our corrected modules
from src.pipeline_builder import build_model_pipelines, get_param_grids
from src.features import validate_feature_data

# Configure paths  
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "insurance_encoded.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Ensure output directories exist
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_DIR / 'model_development.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# CONFIGURATION
TASK = "classification"  # Task type: "classification" for claim approval prediction
TARGET = "claim_status"  # Target column: 0=Approved, 1=Denied
USE_CHARGES_FEATURE = True  # Include charges-based features in training
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_JOBS = -1  # Use all available cores


class ModelDevelopmentPipeline:
    """
    Corrected ML pipeline that prevents CV leakage and ensures train/inference consistency.
    
    Key Improvements:
    - All preprocessing happens INSIDE the Pipeline 
    - GridSearchCV operates on complete Pipelines (no leakage)
    - Saves single Pipeline object (not separate model + scaler)
    - Consistent feature engineering via FeatureBuilder
    """
    
    def __init__(self):
        self.data: pd.DataFrame = None
        self.X_train: pd.DataFrame = None  
        self.X_test: pd.DataFrame = None
        self.y_train: pd.Series = None
        self.y_test: pd.Series = None
        self.pipelines: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.best_pipeline = None
        self.best_model_name: str = ""
        self.best_score: float = -np.inf
        
    def load_and_validate_data(self) -> None:
        """Load and validate input data."""
        logger.info("Loading and validating data...")
        
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
            
        # Load data
        self.data = pd.read_csv(DATA_PATH)
        logger.info(f"Loaded data: {self.data.shape}")
        
        # Validate target column exists
        if TARGET not in self.data.columns:
            raise ValueError(f"Target column '{TARGET}' not found in data")
            
        # Validate feature data
        feature_data = self.data.drop(columns=[TARGET])
        is_valid, error_msg = validate_feature_data(feature_data, require_charges=USE_CHARGES_FEATURE)
        if not is_valid:
            raise ValueError(f"Data validation failed: {error_msg}")
            
        # Log data summary
        target_counts = self.data[TARGET].value_counts()
        logger.info(f"Target distribution: {target_counts.to_dict()}")
        logger.info(f"Features: {list(feature_data.columns)}")
        
    def split_data(self) -> None:
        """Split data into train/test sets with proper stratification."""
        logger.info("Splitting data into train/test sets...")
        
        # Separate features and target
        X = self.data.drop(columns=[TARGET])
        y = self.data[TARGET]
        
        # Stratified split to maintain class distribution
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y  # Maintain class balance
        )
        
        logger.info(f"Train set: {self.X_train.shape}")
        logger.info(f"Test set: {self.X_test.shape}")
        logger.info(f"Train target dist: {self.y_train.value_counts().to_dict()}")
        logger.info(f"Test target dist: {self.y_test.value_counts().to_dict()}")
        
    def build_pipelines(self) -> None:
        """Build complete ML pipelines for different algorithms."""
        logger.info("Building ML pipelines...")
        
        # Build pipelines using our corrected pipeline builder
        self.pipelines = build_model_pipelines(
            task=TASK,
            use_charges_feature=USE_CHARGES_FEATURE
        )
        
        logger.info(f"Built {len(self.pipelines)} pipelines: {list(self.pipelines.keys())}")
        
        # Log pipeline structure for verification
        for name, pipeline in self.pipelines.items():
            steps = [f"{step[0]}:{type(step[1]).__name__}" for step in pipeline.steps]
            logger.debug(f"{name} pipeline: {' -> '.join(steps)}")
            
    def train_and_evaluate_models(self) -> None:
        """Train models using GridSearchCV over complete pipelines (NO LEAKAGE)."""
        logger.info("Training and evaluating models with GridSearchCV...")
        
        # Get hyperparameter grids
        param_grids = get_param_grids(TASK)
        
        # Cross-validation strategy
        cv_strategy = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE
        )
        
        # Train each pipeline
        for model_name, pipeline in self.pipelines.items():
            logger.info(f"Training {model_name}...")
            
            # Get parameter grid for this model
            param_grid = param_grids.get(model_name, {})
            
            try:
                # GridSearchCV over the COMPLETE pipeline (no leakage!)
                grid_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    cv=cv_strategy,
                    scoring='f1',  # F1 score for healthcare (balance precision/recall)
                    n_jobs=N_JOBS,
                    refit=True,
                    verbose=1 if logger.level <= logging.INFO else 0,
                    error_score='raise'  # Fail fast on errors
                )
                
                # Fit the grid search
                grid_search.fit(self.X_train, self.y_train)
                
                # Store results
                cv_score = grid_search.best_score_
                self.results[model_name] = {
                    'pipeline': grid_search.best_estimator_,
                    'cv_score': cv_score,
                    'best_params': grid_search.best_params_,
                    'cv_results': grid_search.cv_results_
                }
                
                # Track best model
                if cv_score > self.best_score:
                    self.best_score = cv_score
                    self.best_model_name = model_name
                    self.best_pipeline = grid_search.best_estimator_
                    
                logger.info(f"{model_name} - CV F1: {cv_score:.4f}, Params: {grid_search.best_params_}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
                
        if self.best_pipeline is None:
            raise RuntimeError("No models successfully trained!")
            
        logger.info(f"Best model: {self.best_model_name} (CV F1: {self.best_score:.4f})")
        
    def evaluate_best_model(self) -> Dict[str, Any]:
        """Evaluate best model on holdout test set."""
        logger.info("Evaluating best model on test set...")
        
        # Make predictions on test set
        y_pred = self.best_pipeline.predict(self.X_test)
        y_pred_proba = None
        
        # Get prediction probabilities if available
        if hasattr(self.best_pipeline, 'predict_proba'):
            y_pred_proba = self.best_pipeline.predict_proba(self.X_test)[:, 1]
            
        # Calculate comprehensive metrics
        test_metrics = {
            'accuracy': float(accuracy_score(self.y_test, y_pred)),
            'precision': float(precision_score(self.y_test, y_pred)),
            'recall': float(recall_score(self.y_test, y_pred)), 
            'f1_score': float(f1_score(self.y_test, y_pred)),
        }
        
        # Add ROC-AUC if probabilities available
        if y_pred_proba is not None:
            test_metrics['roc_auc'] = float(roc_auc_score(self.y_test, y_pred_proba))
            
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        test_metrics['confusion_matrix'] = cm.tolist()
        
        # Log results
        logger.info("=" * 60)
        logger.info("FINAL MODEL PERFORMANCE")
        logger.info("=" * 60)
        logger.info(f"Model: {self.best_model_name}")
        for metric, value in test_metrics.items():
            if metric != 'confusion_matrix':
                logger.info(f"{metric.title()}: {value:.4f}")
                
        logger.info(f"Confusion Matrix:")
        logger.info(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
        logger.info(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
        logger.info("=" * 60)
        
        return test_metrics
        
    def save_artifacts(self, test_metrics: Dict[str, Any]) -> None:
        """Save trained pipeline and results."""
        logger.info("Saving model artifacts...")
        
        # Save the complete trained pipeline (NOT separate model + scaler!)
        model_filename = MODELS_DIR / f"final_pipeline_{TASK}.joblib"
        joblib.dump(self.best_pipeline, model_filename)
        logger.info(f"✅ Saved complete pipeline to: {model_filename}")
        
        # Save comprehensive results
        results_summary = {
            'task': TASK,
            'target': TARGET,
            'use_charges_feature': USE_CHARGES_FEATURE,
            'best_model': self.best_model_name,
            'cv_score': float(self.best_score),
            'test_metrics': test_metrics,
            'model_path': str(model_filename),
            'data_shape': list(self.data.shape),
            'feature_columns': list(self.X_train.columns),
            'training_config': {
                'test_size': TEST_SIZE,
                'cv_folds': CV_FOLDS,
                'random_state': RANDOM_STATE
            }
        }
        
        # Save results as JSON
        results_file = RESULTS_DIR / "model_development_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        logger.info(f"✅ Saved results to: {results_file}")
        
        # Save detailed CV results
        cv_results_file = RESULTS_DIR / "cross_validation_results.json"
        cv_details = {
            name: {
                'cv_score': result['cv_score'],
                'best_params': result['best_params']
            }
            for name, result in self.results.items()
        }
        
        with open(cv_results_file, 'w') as f:
            json.dump(cv_details, f, indent=2)
        logger.info(f"✅ Saved CV results to: {cv_results_file}")
        
    def run_complete_pipeline(self) -> None:
        """Execute the complete corrected ML pipeline."""
        try:
            logger.info("🏥 Starting CORRECTED Model Development Pipeline")
            logger.info("=" * 70)
            
            self.load_and_validate_data()
            self.split_data() 
            self.build_pipelines()
            self.train_and_evaluate_models()
            test_metrics = self.evaluate_best_model()
            self.save_artifacts(test_metrics)
            
            logger.info("✅ Pipeline completed successfully!")
            
            # Final summary
            print("\n" + "🎉" * 20)
            print("MODEL DEVELOPMENT COMPLETED")
            print("🎉" * 20)
            print(f"✅ Best Model: {self.best_model_name}")
            print(f"✅ CV F1-Score: {self.best_score:.4f}")
            print(f"✅ Test F1-Score: {test_metrics['f1_score']:.4f}")
            print(f"✅ Model saved: models/final_pipeline_{TASK}.joblib")
            print("\n🔧 CRITICAL FIXES IMPLEMENTED:")
            print("  - ✅ No CV leakage (preprocessing inside pipelines)")
            print("  - ✅ Proper categorical encoding (OneHotEncoder)")  
            print("  - ✅ Unified preprocessing for all algorithms")
            print("  - ✅ Complete Pipeline saved (not separate pieces)")
            print("  - ✅ Train/inference consistency guaranteed")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Main entry point for model development."""
    
    # Validate configuration
    if TASK not in ["classification", "regression"]:
        raise ValueError(f"Invalid TASK: {TASK}")
        
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
        
    logger.info(f"Configuration:")
    logger.info(f"  Task: {TASK}")
    logger.info(f"  Target: {TARGET}")
    logger.info(f"  Use charges feature: {USE_CHARGES_FEATURE}")
    logger.info(f"  Data path: {DATA_PATH}")
    logger.info(f"  Test size: {TEST_SIZE}")
    logger.info(f"  CV folds: {CV_FOLDS}")
    
    # Run the corrected pipeline
    pipeline = ModelDevelopmentPipeline()
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()