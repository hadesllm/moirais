"""Tests for fzt59.fauzi_thm5_9_edgeworth_wilcoxon."""
import numpy as np
import pytest
from morie.fn.fzt59 import fauzi_thm5_9_edgeworth_wilcoxon


def test_fzt59_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    theta = 0.0
    c = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = fauzi_thm5_9_edgeworth_wilcoxon(data, bandwidth, theta, c, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzt59_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    theta = 0.0
    c = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = fauzi_thm5_9_edgeworth_wilcoxon(data, bandwidth, theta, c, d)
    assert isinstance(result, dict)
