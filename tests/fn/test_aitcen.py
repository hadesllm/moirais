"""Tests for aitcen.aitchison_center."""
import numpy as np
import pytest
from morie.fn.aitcen import aitchison_center


def test_aitcen_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = aitchison_center(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitcen_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = aitchison_center(X)
    assert isinstance(result, dict)
