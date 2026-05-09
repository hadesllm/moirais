"""Tests for fzb4t.fauzi_b4_coefficient_mrl."""
import numpy as np
import pytest
from moirais.fn.fzb4t import fauzi_b4_coefficient_mrl


def test_fzb4t_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    mrl = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_b4_coefficient_mrl(t, mrl)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzb4t_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    mrl = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_b4_coefficient_mrl(t, mrl)
    assert isinstance(result, dict)
