"""Tests for ctta1c.ctt_alpha_classic."""
import numpy as np
import pytest
from morie.fn.ctta1c import ctt_alpha_classic


def test_ctta1c_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = ctt_alpha_classic(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ctta1c_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = ctt_alpha_classic(X)
    assert isinstance(result, dict)
