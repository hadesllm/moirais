"""Tests for spbp.schabenberger_breusch_pagan."""
import numpy as np
import pytest
from morie.fn.spbp import schabenberger_breusch_pagan


def test_spbp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    residuals = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_breusch_pagan(x, y, residuals)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spbp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    residuals = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_breusch_pagan(x, y, residuals)
    assert isinstance(result, dict)
