"""Tests for irtid.irt_identification_constraints."""
import numpy as np
import pytest
from moirais.fn.irtid import irt_identification_constraints


def test_irtid_basic():
    """Test basic functionality."""
    x_init = 0.0
    polarity_idx = np.random.default_rng(42).normal(0, 1, 100)
    pivot_idx = np.random.default_rng(42).normal(0, 1, 100)
    result = irt_identification_constraints(x_init, polarity_idx, pivot_idx)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_irtid_edge():
    """Test edge cases."""
    x_init = 0.0
    polarity_idx = np.random.default_rng(42).normal(0, 1, 100)
    pivot_idx = np.random.default_rng(42).normal(0, 1, 100)
    result = irt_identification_constraints(x_init, polarity_idx, pivot_idx)
    assert isinstance(result, dict)
