"""Tests for rgeqn9b.rangayyan_ch9_ica_ambiguity."""
import numpy as np
import pytest
from moirais.fn.rgeqn9b import rangayyan_ch9_ica_ambiguity


def test_rgeqn9b_basic():
    """Test basic functionality."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    result = rangayyan_ch9_ica_ambiguity(W, s)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgeqn9b_edge():
    """Test edge cases."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    result = rangayyan_ch9_ica_ambiguity(W, s)
    assert isinstance(result, dict)
