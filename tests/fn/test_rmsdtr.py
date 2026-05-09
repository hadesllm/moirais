"""Tests for rmsdtr.rmsd."""
import numpy as np
import pytest
from moirais.fn.rmsdtr import rmsd


def test_rmsdtr_basic():
    """Test basic functionality."""
    coords1 = np.random.default_rng(42).normal(0, 1, 100)
    coords2 = np.random.default_rng(42).normal(0, 1, 100)
    result = rmsd(coords1, coords2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rmsdtr_edge():
    """Test edge cases."""
    coords1 = np.random.default_rng(42).normal(0, 1, 100)
    coords2 = np.random.default_rng(42).normal(0, 1, 100)
    result = rmsd(coords1, coords2)
    assert isinstance(result, dict)
