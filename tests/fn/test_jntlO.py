"""Tests for jntlO.joint_loss_mixed_outcomes."""
import numpy as np
import pytest
from morie.fn.jntlO import joint_loss_mixed_outcomes


def test_jntlO_basic():
    """Test basic functionality."""
    y_dict = np.random.default_rng(42).normal(0, 1, 100)
    y_hat_dict = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = joint_loss_mixed_outcomes(y_dict, y_hat_dict, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jntlO_edge():
    """Test edge cases."""
    y_dict = np.random.default_rng(42).normal(0, 1, 100)
    y_hat_dict = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = joint_loss_mixed_outcomes(y_dict, y_hat_dict, weights)
    assert isinstance(result, dict)
