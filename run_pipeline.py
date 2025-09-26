#!/usr/bin/env python3
"""
Simple machine learning pipeline for health insurance claim prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json

def load_data():
    """Load the sample dataset."""
    data_path = Path("data/insurance_sample.csv")
    if not data_path.exists():
        # Create sample data if it doesn't exist
        create_sample_data()

    df = pd.read_csv(data_path)
    return df

def create_sample_data():
    """Create a sample dataset for training."""
    np.random.seed(42)
    n_samples = 399

    data = {
        'age': np.random.randint(18, 65, n_samples),
        'sex': np.random.choice([0, 1], n_samples),
        'bmi': np.random.normal(26, 4, n_samples).clip(15, 50),
        'children': np.random.choice([0, 1, 2, 3, 4], n_samples),
        'smoker': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'region': np.random.choice([0, 1, 2, 3], n_samples)
    }

    # Create target based on logical rules
    target = []
    for i in range(n_samples):
        # Higher probability of approval for non-smokers with normal BMI
        prob = 0.7
        if data['smoker'][i] == 1:
            prob -= 0.3
        if data['bmi'][i] > 30:
            prob -= 0.2
        if data['age'][i] > 50:
            prob -= 0.1

        target.append(1 if np.random.random() < prob else 0)

    data['claim_status'] = target

    df = pd.DataFrame(data)

    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/insurance_sample.csv", index=False)
    print(f"Created sample dataset with {len(df)} records")

    return df

def build_models():
    """Build and train machine learning models."""
    # Load data
    df = load_data()

    # Prepare features and target
    X = df.drop('claim_status', axis=1)
    y = df['claim_status']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize models
    models = {
        'logistic_regression': LogisticRegression(random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    # Train and evaluate models
    for name, model in models.items():
        print(f"Training {name}...")

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }

        print(f"{name.replace('_', ' ').title()} Accuracy: {accuracy:.3f}")

    return results, X_test, y_test

def save_models(results):
    """Save trained models."""
    # Create models directory
    Path("models").mkdir(exist_ok=True)

    for name, result in results.items():
        model_path = f"models/{name}.joblib"
        joblib.dump(result['model'], model_path)
        print(f"Saved {name} to {model_path}")

def create_plots(results, X_test, y_test):
    """Create visualization plots."""
    # Create plots directory
    Path("plots").mkdir(exist_ok=True)

    # Model comparison plot
    plt.figure(figsize=(10, 6))

    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]

    plt.subplot(1, 2, 1)
    plt.bar([m.replace('_', ' ').title() for m in models], accuracies)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    # Feature importance plot (Random Forest)
    if 'random_forest' in results:
        plt.subplot(1, 2, 2)
        rf_model = results['random_forest']['model']
        feature_names = ['Age', 'Sex', 'BMI', 'Children', 'Smoker', 'Region']
        importances = rf_model.feature_importances_

        plt.bar(feature_names, importances)
        plt.title('Feature Importance (Random Forest)')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Feature importance plot
    if 'random_forest' in results:
        plt.figure(figsize=(8, 6))
        rf_model = results['random_forest']['model']
        feature_names = ['Age', 'Sex', 'BMI', 'Children', 'Smoker', 'Region']
        importances = rf_model.feature_importances_

        indices = np.argsort(importances)[::-1]

        plt.bar(range(len(importances)), importances[indices])
        plt.title('Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)

        plt.tight_layout()
        plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

    print("Plots saved to plots/ directory")

def save_results_artifacts(results, best_model_name, best_accuracy):
    """Save results as CSV and JSON files."""
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Create metrics summary CSV
    metrics_data = []
    for model_name, model_data in results.items():
        metrics_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': f"{model_data['accuracy']:.3f}",
            'Best_Model': 'Yes' if model_name == best_model_name else 'No'
        })

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv('results/metrics_summary.csv', index=False)

    # Create detailed metrics JSON
    metrics_json = {
        'best_model': best_model_name.replace('_', ' ').title(),
        'best_accuracy': float(f"{best_accuracy:.3f}"),
        'models': {}
    }

    for model_name, model_data in results.items():
        metrics_json['models'][model_name] = {
            'accuracy': float(f"{model_data['accuracy']:.3f}"),
            'display_name': model_name.replace('_', ' ').title()
        }

    with open('results/metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)

    print("Results artifacts saved to results/ directory")

def main():
    """Main pipeline execution."""
    print("Health Insurance Claim Prediction Pipeline")
    print("=" * 50)

    # Build and train models
    results, X_test, y_test = build_models()

    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model_name]['accuracy']

    print(f"\nBest model: {best_model_name.replace('_', ' ').title()}")
    print(f"Best accuracy: {best_accuracy:.3f}")

    # Save models
    save_models(results)

    # Create plots
    create_plots(results, X_test, y_test)

    # Save results artifacts
    save_results_artifacts(results, best_model_name, best_accuracy)

    print("\nPipeline completed successfully!")
    print(f"Models saved to models/ directory")
    print(f"Plots saved to plots/ directory")
    print(f"Results saved to results/ directory")

if __name__ == "__main__":
    main()