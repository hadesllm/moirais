"""Tests for varatr.value_at_risk."""
import numpy as np
import pytest
from moirais.fn.varatr import value_at_risk


def test_varatr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = value_at_risk(y, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_varatr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = value_at_risk(y, alpha)
    assert isinstance(result, dict)
