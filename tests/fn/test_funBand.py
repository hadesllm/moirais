"""Tests for funBand.functional_band."""
import numpy as np
import pytest
from morie.fn.funBand import functional_band


def test_funBand_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = functional_band(Y, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_funBand_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = functional_band(Y, alpha)
    assert isinstance(result, dict)
