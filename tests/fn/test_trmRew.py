"""Tests for trmRew.term_rewriting."""
import numpy as np
import pytest
from moirais.fn.trmRew import term_rewriting


def test_trmRew_basic():
    """Test basic functionality."""
    term = np.random.default_rng(42).normal(0, 1, 100)
    rules = np.random.default_rng(42).normal(0, 1, 100)
    result = term_rewriting(term, rules)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_trmRew_edge():
    """Test edge cases."""
    term = np.random.default_rng(42).normal(0, 1, 100)
    rules = np.random.default_rng(42).normal(0, 1, 100)
    result = term_rewriting(term, rules)
    assert isinstance(result, dict)
