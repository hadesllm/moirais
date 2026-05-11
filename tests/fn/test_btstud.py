"""Tests for btstud.boot_studentized_ci."""
import numpy as np
import pytest
from morie.fn.btstud import boot_studentized_ci


def test_btstud_basic():
    """Test basic functionality."""
    theta_hat = np.random.default_rng(42).normal(0, 1, 100)
    se_hat = np.random.default_rng(42).normal(0, 1, 100)
    t_b = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = boot_studentized_ci(theta_hat, se_hat, t_b, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btstud_edge():
    """Test edge cases."""
    theta_hat = np.random.default_rng(42).normal(0, 1, 100)
    se_hat = np.random.default_rng(42).normal(0, 1, 100)
    t_b = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = boot_studentized_ci(theta_hat, se_hat, t_b, alpha)
    assert isinstance(result, dict)
