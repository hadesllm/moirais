"""Tests for otmxh.ot_mixture_w2."""
import numpy as np
import pytest
from morie.fn.otmxh import ot_mixture_w2


def test_otmxh_basic():
    """Test basic functionality."""
    mus1 = np.random.default_rng(42).normal(0, 1, 100)
    Sigmas1 = np.random.default_rng(42).normal(0, 1, 100)
    w1 = np.random.default_rng(42).normal(0, 1, 100)
    mus2 = np.random.default_rng(42).normal(0, 1, 100)
    Sigmas2 = np.random.default_rng(42).normal(0, 1, 100)
    w2 = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_mixture_w2(mus1, Sigmas1, w1, mus2, Sigmas2, w2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otmxh_edge():
    """Test edge cases."""
    mus1 = np.random.default_rng(42).normal(0, 1, 100)
    Sigmas1 = np.random.default_rng(42).normal(0, 1, 100)
    w1 = np.random.default_rng(42).normal(0, 1, 100)
    mus2 = np.random.default_rng(42).normal(0, 1, 100)
    Sigmas2 = np.random.default_rng(42).normal(0, 1, 100)
    w2 = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_mixture_w2(mus1, Sigmas1, w1, mus2, Sigmas2, w2)
    assert isinstance(result, dict)
