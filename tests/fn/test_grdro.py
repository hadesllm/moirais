"""Tests for grdro.geron_dropout."""
import numpy as np
import pytest
from morie.fn.grdro import geron_dropout


def test_grdro_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    p = 5
    seed = 42
    result = geron_dropout(a, p, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grdro_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    p = 5
    seed = 42
    result = geron_dropout(a, p, seed)
    assert isinstance(result, dict)
