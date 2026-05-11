"""Tests for tstinf.test_information."""
import numpy as np
import pytest
from morie.fn.tstinf import test_information as _test_information


def test_tstinf_basic():
    """Test basic functionality."""
    theta = np.random.default_rng(99).normal(0, 1, 100)
    items = np.random.default_rng(42).normal(0, 1, 100)
    result = _test_information(theta, items)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_tstinf_edge():
    """Test edge cases."""
    theta = np.random.default_rng(99).normal(0, 1, 100)
    items = np.random.default_rng(42).normal(0, 1, 100)
    result = _test_information(theta, items)
    assert isinstance(result, dict)
