#!/usr/bin/env python3
"""
Basic tests for the Flask web application.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def test_app_import():
    """Test that the app module can be imported."""
    try:
        import app
        assert hasattr(app, 'app')
    except ImportError:
        pytest.fail("Could not import app module")

def test_routes_exist():
    """Test that main routes are defined."""
    import app

    # Get all route rules
    routes = [rule.rule for rule in app.app.url_map.iter_rules()]

    # Check main routes exist
    assert '/' in routes
    assert '/predict' in routes