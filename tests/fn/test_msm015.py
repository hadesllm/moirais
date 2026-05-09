"""Tests for msm015.mvsml_linear_mixed_models_eq_5_3."""
import numpy as np
import pytest
from moirais.fn.msm015 import mvsml_linear_mixed_models_eq_5_3


def test_msm015_basic():
    """Test basic functionality."""
    derived = np.random.default_rng(42).normal(0, 1, 100)
    re = np.random.default_rng(42).normal(0, 1, 100)
    ectance = np.random.default_rng(42).normal(0, 1, 100)
    information = np.random.default_rng(42).normal(0, 1, 100)
    Krause = np.random.default_rng(42).normal(0, 1, 100)
    et = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_3(derived, re, ectance, information, Krause, et)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm015_edge():
    """Test edge cases."""
    derived = np.random.default_rng(42).normal(0, 1, 100)
    re = np.random.default_rng(42).normal(0, 1, 100)
    ectance = np.random.default_rng(42).normal(0, 1, 100)
    information = np.random.default_rng(42).normal(0, 1, 100)
    Krause = np.random.default_rng(42).normal(0, 1, 100)
    et = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_3(derived, re, ectance, information, Krause, et)
    assert isinstance(result, dict)
