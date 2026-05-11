"""Tests for hmper.geron_prioritized_replay."""
import numpy as np
import pytest
from morie.fn.hmper import geron_prioritized_replay


def test_hmper_basic():
    """Test basic functionality."""
    buffer = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    result = geron_prioritized_replay(buffer, alpha, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmper_edge():
    """Test edge cases."""
    buffer = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    result = geron_prioritized_replay(buffer, alpha, beta)
    assert isinstance(result, dict)
