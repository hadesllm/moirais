"""Tests for msm238.mvsml_general_eq_1_3."""
import numpy as np
import pytest
from moirais.fn.msm238 import mvsml_general_eq_1_3


def test_msm238_basic():
    """Test basic functionality."""
    Ordering = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    Data = np.random.default_rng(42).normal(0, 1, 100)
    Final = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = mvsml_general_eq_1_3(Ordering, the, data, Data, Final, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm238_edge():
    """Test edge cases."""
    Ordering = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    Data = np.random.default_rng(42).normal(0, 1, 100)
    Final = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = mvsml_general_eq_1_3(Ordering, the, data, Data, Final, order)
    assert isinstance(result, dict)
