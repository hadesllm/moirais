"""Tests for kabsch.kabsch_superpose."""
import numpy as np
import pytest
from morie.fn.kabsch import kabsch_superpose


def test_kabsch_basic():
    """Test basic functionality."""
    coords1 = np.random.default_rng(42).normal(0, 1, 100)
    coords2 = np.random.default_rng(42).normal(0, 1, 100)
    result = kabsch_superpose(coords1, coords2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kabsch_edge():
    """Test edge cases."""
    coords1 = np.random.default_rng(42).normal(0, 1, 100)
    coords2 = np.random.default_rng(42).normal(0, 1, 100)
    result = kabsch_superpose(coords1, coords2)
    assert isinstance(result, dict)
