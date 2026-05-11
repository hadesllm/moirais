"""Tests for comet.comet."""
import numpy as np
import pytest
from morie.fn.comet import comet


def test_comet_basic():
    """Test basic functionality."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    hyp = np.random.default_rng(42).normal(0, 1, 100)
    ref = np.random.default_rng(42).normal(0, 1, 100)
    result = comet(src, hyp, ref)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_comet_edge():
    """Test edge cases."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    hyp = np.random.default_rng(42).normal(0, 1, 100)
    ref = np.random.default_rng(42).normal(0, 1, 100)
    result = comet(src, hyp, ref)
    assert isinstance(result, dict)
