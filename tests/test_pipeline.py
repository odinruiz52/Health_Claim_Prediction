#!/usr/bin/env python3
"""
Basic tests for the machine learning pipeline.
"""

import pytest
from pathlib import Path

def test_training_script_exists():
    """Test that the main training script exists."""
    script_path = Path(__file__).parent.parent / "run_pipeline.py"
    assert script_path.exists(), "run_pipeline.py should exist"

def test_models_directory_exists():
    """Test that models directory exists."""
    models_dir = Path(__file__).parent.parent / "models"
    assert models_dir.exists(), "models directory should exist"