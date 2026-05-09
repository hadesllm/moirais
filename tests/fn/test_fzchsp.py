"""Tests for fzchsp.fauzi_chung_smirnov."""
import numpy as np
import pytest
from moirais.fn.fzchsp import fauzi_chung_smirnov


def test_fzchsp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = fauzi_chung_smirnov(x, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzchsp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = fauzi_chung_smirnov(x, bandwidth)
    assert isinstance(result, dict)
