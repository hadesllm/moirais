"""Tests for otmnge.ot_marginal_negent."""
import numpy as np
import pytest
from moirais.fn.otmnge import ot_marginal_negent


def test_otmnge_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = ot_marginal_negent(T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otmnge_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = ot_marginal_negent(T)
    assert isinstance(result, dict)
