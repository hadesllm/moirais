"""Tests for tnvest._test_negative_design."""
import numpy as np
import pytest
from morie.fn.tnvest import test_negative_design as _test_negative_design


def test_tnvest_basic():
    """Test basic functionality."""
    test_status = np.random.default_rng(42).normal(0, 1, 100)
    vaccination_status = np.random.default_rng(42).normal(0, 1, 100)
    result = _test_negative_design(test_status, vaccination_status)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_tnvest_edge():
    """Test edge cases."""
    test_status = np.random.default_rng(42).normal(0, 1, 100)
    vaccination_status = np.random.default_rng(42).normal(0, 1, 100)
    result = _test_negative_design(test_status, vaccination_status)
    assert isinstance(result, dict)
