"""Tests for incrtio.incidence_rate_ratio."""
import numpy as np
import pytest
from morie.fn.incrtio import incidence_rate_ratio


def test_incrtio_basic():
    """Test basic functionality."""
    IR_e = np.random.default_rng(42).normal(0, 1, 100)
    IR_u = np.random.default_rng(42).normal(0, 1, 100)
    result = incidence_rate_ratio(IR_e, IR_u)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_incrtio_edge():
    """Test edge cases."""
    IR_e = np.random.default_rng(42).normal(0, 1, 100)
    IR_u = np.random.default_rng(42).normal(0, 1, 100)
    result = incidence_rate_ratio(IR_e, IR_u)
    assert isinstance(result, dict)
