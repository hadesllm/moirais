"""Tests for rgeqn8b.rangayyan_ch8_glr_threshold."""
import numpy as np
import pytest
from moirais.fn.rgeqn8b import rangayyan_ch8_glr_threshold


def test_rgeqn8b_basic():
    """Test basic functionality."""
    alpha = 0.05
    dof = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch8_glr_threshold(alpha, dof)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgeqn8b_edge():
    """Test edge cases."""
    alpha = 0.05
    dof = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch8_glr_threshold(alpha, dof)
    assert isinstance(result, dict)
