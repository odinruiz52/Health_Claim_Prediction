#!/usr/bin/env python3
"""
Basic tests for file I/O and metrics.
"""

import pytest
from pathlib import Path

def test_data_file_exists():
    """Test that sample data file exists."""
    data_path = Path(__file__).parent.parent / "data" / "insurance_sample.csv"
    assert data_path.exists(), "Sample data file should exist"

def test_requirements_file_exists():
    """Test that requirements.txt exists."""
    req_path = Path(__file__).parent.parent / "requirements.txt"
    assert req_path.exists(), "requirements.txt should exist"