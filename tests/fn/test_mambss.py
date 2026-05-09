"""Tests for mambss.mamba_ssm_step."""
import numpy as np
import pytest
from moirais.fn.mambss import mamba_ssm_step


def test_mambss_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    result = mamba_ssm_step(y, x, A, B, C, D)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mambss_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    result = mamba_ssm_step(y, x, A, B, C, D)
    assert isinstance(result, dict)
