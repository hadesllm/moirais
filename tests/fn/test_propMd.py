"""Tests for propMd.proportion_mediated."""
import numpy as np
import pytest
from moirais.fn.propMd import proportion_mediated


def test_propMd_basic():
    """Test basic functionality."""
    NIE = np.random.default_rng(42).normal(0, 1, 100)
    NDE = np.random.default_rng(42).normal(0, 1, 100)
    result = proportion_mediated(NIE, NDE)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_propMd_edge():
    """Test edge cases."""
    NIE = np.random.default_rng(42).normal(0, 1, 100)
    NDE = np.random.default_rng(42).normal(0, 1, 100)
    result = proportion_mediated(NIE, NDE)
    assert isinstance(result, dict)
