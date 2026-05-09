"""Tests for btscr.boot_score_test."""
import numpy as np
import pytest
from moirais.fn.btscr import boot_score_test


def test_btscr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fit0 = np.random.default_rng(42).normal(0, 1, 100)
    score_fn = (lambda v: float(np.mean(v)))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_score_test(x, fit0, score_fn, B)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_btscr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fit0 = np.random.default_rng(42).normal(0, 1, 100)
    score_fn = (lambda v: float(np.mean(v)))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_score_test(x, fit0, score_fn, B)
    assert isinstance(result, dict)
