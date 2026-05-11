"""Tests for hmmxp2.geron_mixed_precision."""
import numpy as np
import pytest
from morie.fn.hmmxp2 import geron_mixed_precision


def test_hmmxp2_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    loss_scale = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_mixed_precision(model, loss_scale)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmxp2_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    loss_scale = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_mixed_precision(model, loss_scale)
    assert isinstance(result, dict)
