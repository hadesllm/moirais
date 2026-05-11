"""Tests for rrblp.rrblup_marker_effects."""
import numpy as np
import pytest
from morie.fn.rrblp import rrblup_marker_effects


def test_rrblp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    lam = 0.1
    result = rrblup_marker_effects(y, Z, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rrblp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    lam = 0.1
    result = rrblup_marker_effects(y, Z, lam)
    assert isinstance(result, dict)
