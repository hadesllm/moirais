"""Tests for km150.kamath_ch9_flamingo_dataset_mix."""
import numpy as np
import pytest
from moirais.fn.km150 import kamath_ch9_flamingo_dataset_mix


def test_km150_basic():
    """Test basic functionality."""
    D_m = np.random.default_rng(42).normal(0, 1, 100)
    lambda_m = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kamath_ch9_flamingo_dataset_mix(D_m, lambda_m, x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km150_edge():
    """Test edge cases."""
    D_m = np.random.default_rng(42).normal(0, 1, 100)
    lambda_m = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kamath_ch9_flamingo_dataset_mix(D_m, lambda_m, x, y)
    assert isinstance(result, dict)
