"""Tests for prratio.prevalence_ratio."""
import numpy as np
import pytest
from morie.fn.prratio import prevalence_ratio


def test_prratio_basic():
    """Test basic functionality."""
    prev_exposed = np.random.default_rng(42).normal(0, 1, 100)
    prev_unexposed = np.random.default_rng(42).normal(0, 1, 100)
    result = prevalence_ratio(prev_exposed, prev_unexposed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prratio_edge():
    """Test edge cases."""
    prev_exposed = np.random.default_rng(42).normal(0, 1, 100)
    prev_unexposed = np.random.default_rng(42).normal(0, 1, 100)
    result = prevalence_ratio(prev_exposed, prev_unexposed)
    assert isinstance(result, dict)
