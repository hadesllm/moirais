"""Tests for hmrvn.geron_revnet."""
import numpy as np
import pytest
from moirais.fn.hmrvn import geron_revnet


def test_hmrvn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    G = np.eye(10)
    result = geron_revnet(x, F, G)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmrvn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    G = np.eye(10)
    result = geron_revnet(x, F, G)
    assert isinstance(result, dict)
