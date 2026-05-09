"""Tests for mtsbnd.monotone_treatment_selection."""
import numpy as np
import pytest
from moirais.fn.mtsbnd import monotone_treatment_selection


def test_mtsbnd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    y_min = 0
    y_max = 100
    result = monotone_treatment_selection(y, D, y_min, y_max)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mtsbnd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    y_min = 0
    y_max = 100
    result = monotone_treatment_selection(y, D, y_min, y_max)
    assert isinstance(result, dict)
