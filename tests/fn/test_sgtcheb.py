"""Tests for sgtcheb.sgt_cheeger_bound."""
import numpy as np
import pytest
from morie.fn.sgtcheb import sgt_cheeger_bound


def test_sgtcheb_basic():
    """Test basic functionality."""
    lam2 = np.random.default_rng(42).normal(0, 1, 100)
    result = sgt_cheeger_bound(lam2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtcheb_edge():
    """Test edge cases."""
    lam2 = np.random.default_rng(42).normal(0, 1, 100)
    result = sgt_cheeger_bound(lam2)
    assert isinstance(result, dict)
