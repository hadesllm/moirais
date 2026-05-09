"""Tests for msm003.mvsml_general_eq_1_3."""
import numpy as np
import pytest
from moirais.fn.msm003 import mvsml_general_eq_1_3


def test_msm003_basic():
    """Test basic functionality."""
    environment = np.random.default_rng(42).normal(0, 1, 100)
    which = np.random.default_rng(42).normal(0, 1, 100)
    can = np.random.default_rng(42).normal(0, 1, 100)
    be = np.random.default_rng(42).normal(0, 1, 100)
    attributed = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_3(environment, which, can, be, attributed, to)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm003_edge():
    """Test edge cases."""
    environment = np.random.default_rng(42).normal(0, 1, 100)
    which = np.random.default_rng(42).normal(0, 1, 100)
    can = np.random.default_rng(42).normal(0, 1, 100)
    be = np.random.default_rng(42).normal(0, 1, 100)
    attributed = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_3(environment, which, can, be, attributed, to)
    assert isinstance(result, dict)
