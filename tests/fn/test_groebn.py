"""Tests for groebn.groebner."""
import numpy as np
import pytest
from moirais.fn.groebn import groebner


def test_groebn_basic():
    """Test basic functionality."""
    polys = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = groebner(polys, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_groebn_edge():
    """Test edge cases."""
    polys = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = groebner(polys, order)
    assert isinstance(result, dict)
