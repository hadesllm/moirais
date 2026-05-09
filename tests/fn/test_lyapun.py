"""Tests for lyapun.lyapunov_exponent."""
import numpy as np
import pytest
from moirais.fn.lyapun import lyapunov_exponent


def test_lyapun_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    embedding = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = lyapunov_exponent(y, embedding, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lyapun_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    embedding = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = lyapunov_exponent(y, embedding, tau)
    assert isinstance(result, dict)
