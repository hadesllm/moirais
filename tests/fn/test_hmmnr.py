"""Tests for hmmnr.geron_max_norm."""
import numpy as np
import pytest
from moirais.fn.hmmnr import geron_max_norm


def test_hmmnr_basic():
    """Test basic functionality."""
    w = np.random.default_rng(45).exponential(1, 100)
    r = 10
    result = geron_max_norm(w, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmnr_edge():
    """Test edge cases."""
    w = np.random.default_rng(45).exponential(1, 100)
    r = 10
    result = geron_max_norm(w, r)
    assert isinstance(result, dict)
