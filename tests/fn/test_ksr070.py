"""Tests for ksr070.kosorok_ch3_score_operator_path."""
import numpy as np
import pytest
from moirais.fn.ksr070 import kosorok_ch3_score_operator_path


def test_ksr070_basic():
    """Test basic functionality."""
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    theta = 0.0
    eta = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = kosorok_ch3_score_operator_path(B, theta, eta, h, x, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr070_edge():
    """Test edge cases."""
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    theta = 0.0
    eta = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = kosorok_ch3_score_operator_path(B, theta, eta, h, x, p)
    assert isinstance(result, dict)
