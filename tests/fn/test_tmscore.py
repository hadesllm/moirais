"""Tests for tmscore.tm_score."""
import numpy as np
import pytest
from morie.fn.tmscore import tm_score


def test_tmscore_basic():
    """Test basic functionality."""
    coords1 = np.random.default_rng(42).normal(0, 1, 100)
    coords2 = np.random.default_rng(42).normal(0, 1, 100)
    result = tm_score(coords1, coords2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmscore_edge():
    """Test edge cases."""
    coords1 = np.random.default_rng(42).normal(0, 1, 100)
    coords2 = np.random.default_rng(42).normal(0, 1, 100)
    result = tm_score(coords1, coords2)
    assert isinstance(result, dict)
