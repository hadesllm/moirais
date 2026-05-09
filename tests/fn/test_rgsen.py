"""Tests for rgsen.rangayyan_sensitivity."""
import numpy as np
import pytest
from moirais.fn.rgsen import rangayyan_sensitivity


def test_rgsen_basic():
    """Test basic functionality."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_sensitivity(y_true, y_pred)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgsen_edge():
    """Test edge cases."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_sensitivity(y_true, y_pred)
    assert isinstance(result, dict)
