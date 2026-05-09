"""Tests for engrgr.engle_granger."""
import numpy as np
import pytest
from moirais.fn.engrgr import engle_granger


def test_engrgr_basic():
    """Test basic functionality."""
    Y1 = np.random.default_rng(42).normal(0, 1, 100)
    Y2 = np.random.default_rng(42).normal(0, 1, 100)
    result = engle_granger(Y1, Y2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_engrgr_edge():
    """Test edge cases."""
    Y1 = np.random.default_rng(42).normal(0, 1, 100)
    Y2 = np.random.default_rng(42).normal(0, 1, 100)
    result = engle_granger(Y1, Y2)
    assert isinstance(result, dict)
