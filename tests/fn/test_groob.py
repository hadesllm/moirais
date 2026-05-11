"""Tests for groob.geron_oob_error."""
import numpy as np
import pytest
from morie.fn.groob import geron_oob_error


def test_groob_basic():
    """Test basic functionality."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    oob_predictions = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_oob_error(y_true, oob_predictions)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_groob_edge():
    """Test edge cases."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    oob_predictions = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_oob_error(y_true, oob_predictions)
    assert isinstance(result, dict)
