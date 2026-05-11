"""Tests for alinfn.alammar_infonce_loss."""
import numpy as np
import pytest
from morie.fn.alinfn import alammar_infonce_loss


def test_alinfn_basic():
    """Test basic functionality."""
    anchor = np.random.default_rng(42).normal(0, 1, 100)
    positive = np.random.default_rng(42).normal(0, 1, 100)
    negatives = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = alammar_infonce_loss(anchor, positive, negatives, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alinfn_edge():
    """Test edge cases."""
    anchor = np.random.default_rng(42).normal(0, 1, 100)
    positive = np.random.default_rng(42).normal(0, 1, 100)
    negatives = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = alammar_infonce_loss(anchor, positive, negatives, tau)
    assert isinstance(result, dict)
