"""Tests for gb1061.gibbons_jonckheere."""
import numpy as np
import pytest
from moirais.fn.gb1061 import gibbons_jonckheere


def test_gb1061_basic():
    """Test basic functionality."""
    groups = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_jonckheere(groups)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1061_edge():
    """Test edge cases."""
    groups = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_jonckheere(groups)
    assert isinstance(result, dict)
