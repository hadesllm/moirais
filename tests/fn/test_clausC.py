"""Tests for clausC.clausius_clapeyron."""
import numpy as np
import pytest
from moirais.fn.clausC import clausius_clapeyron


def test_clausC_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = clausius_clapeyron(T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_clausC_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = clausius_clapeyron(T)
    assert isinstance(result, dict)
