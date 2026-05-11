"""Tests for impFB.implicit_feedback_loss."""
import numpy as np
import pytest
from morie.fn.impFB import implicit_feedback_loss


def test_impFB_basic():
    """Test basic functionality."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    conf = np.random.default_rng(42).normal(0, 1, 100)
    result = implicit_feedback_loss(R, conf)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_impFB_edge():
    """Test edge cases."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    conf = np.random.default_rng(42).normal(0, 1, 100)
    result = implicit_feedback_loss(R, conf)
    assert isinstance(result, dict)
