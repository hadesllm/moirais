"""Tests for otentr.ot_entropy_regulariser."""
import numpy as np
import pytest
from moirais.fn.otentr import ot_entropy_regulariser


def test_otentr_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    epsilon = 1e-6
    result = ot_entropy_regulariser(T, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otentr_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    epsilon = 1e-6
    result = ot_entropy_regulariser(T, epsilon)
    assert isinstance(result, dict)
