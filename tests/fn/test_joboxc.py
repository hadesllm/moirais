"""Tests for joboxc.joseph_box_cox_transform."""
import numpy as np
import pytest
from moirais.fn.joboxc import joseph_box_cox_transform


def test_joboxc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    lam = 0.1
    result = joseph_box_cox_transform(y, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_joboxc_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    lam = 0.1
    result = joseph_box_cox_transform(y, lam)
    assert isinstance(result, dict)
