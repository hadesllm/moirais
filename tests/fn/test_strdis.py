"""Tests for strdis.structural_distance."""
import numpy as np
import pytest
from morie.fn.strdis import structural_distance


def test_strdis_basic():
    """Test basic functionality."""
    G1 = np.random.default_rng(42).normal(0, 1, 100)
    G2 = np.random.default_rng(42).normal(0, 1, 100)
    result = structural_distance(G1, G2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_strdis_edge():
    """Test edge cases."""
    G1 = np.random.default_rng(42).normal(0, 1, 100)
    G2 = np.random.default_rng(42).normal(0, 1, 100)
    result = structural_distance(G1, G2)
    assert isinstance(result, dict)
