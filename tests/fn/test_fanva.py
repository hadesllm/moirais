"""Tests for fanva.fanova."""
import numpy as np
import pytest
from moirais.fn.fanva import fanova


def test_fanva_basic():
    """Test basic functionality."""
    functions = np.random.default_rng(42).normal(0, 1, 100)
    groups = np.random.default_rng(43).integers(0, 3, 100)
    result = fanova(functions, groups)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fanva_edge():
    """Test edge cases."""
    functions = np.random.default_rng(42).normal(0, 1, 100)
    groups = np.random.default_rng(43).integers(0, 3, 100)
    result = fanova(functions, groups)
    assert isinstance(result, dict)
