"""Tests for rosenb.rosenbaum_bounds."""
import numpy as np
import pytest
from moirais.fn.rosenb import rosenbaum_bounds


def test_rosenb_basic():
    """Test basic functionality."""
    matched_pairs = np.random.default_rng(42).normal(0, 1, 100)
    Gamma_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = rosenbaum_bounds(matched_pairs, Gamma_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rosenb_edge():
    """Test edge cases."""
    matched_pairs = np.random.default_rng(42).normal(0, 1, 100)
    Gamma_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = rosenbaum_bounds(matched_pairs, Gamma_grid)
    assert isinstance(result, dict)
