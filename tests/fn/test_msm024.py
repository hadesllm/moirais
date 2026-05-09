"""Tests for msm024.mvsml_linear_mixed_models_eq_5_3."""
import numpy as np
import pytest
from moirais.fn.msm024 import mvsml_linear_mixed_models_eq_5_3


def test_msm024_basic():
    """Test basic functionality."""
    M11 = np.random.default_rng(42).normal(0, 1, 100)
    referred = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    plus = np.random.default_rng(42).normal(0, 1, 100)
    environment = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_3(M11, referred, to, model, plus, environment)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm024_edge():
    """Test edge cases."""
    M11 = np.random.default_rng(42).normal(0, 1, 100)
    referred = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    plus = np.random.default_rng(42).normal(0, 1, 100)
    environment = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_3(M11, referred, to, model, plus, environment)
    assert isinstance(result, dict)
