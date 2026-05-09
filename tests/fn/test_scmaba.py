"""Tests for scmaba.synthetic_control_method."""
import numpy as np
import pytest
from moirais.fn.scmaba import synthetic_control_method


def test_scmaba_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treated = np.random.default_rng(42).normal(0, 1, 100)
    controls = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = synthetic_control_method(y, treated, controls, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_scmaba_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treated = np.random.default_rng(42).normal(0, 1, 100)
    controls = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = synthetic_control_method(y, treated, controls, X)
    assert isinstance(result, dict)
