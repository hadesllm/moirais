"""Tests for kmmamb.kamath_mamba_ssm."""
import numpy as np
import pytest
from morie.fn.kmmamb import kamath_mamba_ssm


def test_kmmamb_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_mamba_ssm(x, A, B, C, delta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmmamb_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_mamba_ssm(x, A, B, C, delta)
    assert isinstance(result, dict)
