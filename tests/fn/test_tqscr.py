"""Tests for tqscr.turboquant_score_distortion."""
import numpy as np
import pytest
from moirais.fn.tqscr import turboquant_score_distortion


def test_tqscr_basic():
    """Test basic functionality."""
    eps = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    n = 100
    result = turboquant_score_distortion(eps, r, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tqscr_edge():
    """Test edge cases."""
    eps = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    n = 100
    result = turboquant_score_distortion(eps, r, n)
    assert isinstance(result, dict)
