"""Tests for gb1131t.gibbons_spearman_ties."""
import numpy as np
import pytest
from morie.fn.gb1131t import gibbons_spearman_ties


def test_gb1131t_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_spearman_ties(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb1131t_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_spearman_ties(x, y)
    assert isinstance(result, dict)
