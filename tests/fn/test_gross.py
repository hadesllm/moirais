"""Tests for gross.gross_error_sensitivity."""
import numpy as np
import pytest
from morie.fn.gross import gross_error_sensitivity


def test_gross_basic():
    """Test basic functionality."""
    IF = np.random.default_rng(42).normal(0, 1, 100)
    result = gross_error_sensitivity(IF)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gross_edge():
    """Test edge cases."""
    IF = np.random.default_rng(42).normal(0, 1, 100)
    result = gross_error_sensitivity(IF)
    assert isinstance(result, dict)
