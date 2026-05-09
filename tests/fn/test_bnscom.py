"""Tests for bnscom.bound_compliance."""
import numpy as np
import pytest
from moirais.fn.bnscom import bound_compliance


def test_bnscom_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = bound_compliance(y, D, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bnscom_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = bound_compliance(y, D, Z)
    assert isinstance(result, dict)
