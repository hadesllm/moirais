"""Tests for msm019.mvsml_linear_mixed_models_eq_5_1."""
import numpy as np
import pytest
from moirais.fn.msm019 import mvsml_linear_mixed_models_eq_5_1


def test_msm019_basic():
    """Test basic functionality."""
    The = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    code = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    reproduce = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_1(The, R, code, to, reproduce, this)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm019_edge():
    """Test edge cases."""
    The = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    code = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    reproduce = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_1(The, R, code, to, reproduce, this)
    assert isinstance(result, dict)
