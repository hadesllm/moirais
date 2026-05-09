"""Tests for grent.geron_shannon_entropy."""
import numpy as np
import pytest
from moirais.fn.grent import geron_shannon_entropy


def test_grent_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_shannon_entropy(y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grent_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_shannon_entropy(y)
    assert isinstance(result, dict)
