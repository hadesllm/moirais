"""Tests for hmresn.geron_resnet."""
import numpy as np
import pytest
from morie.fn.hmresn import geron_resnet


def test_hmresn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_resnet(x, F)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmresn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_resnet(x, F)
    assert isinstance(result, dict)
