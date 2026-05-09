"""Tests for reldge.reliability_gebv."""
import numpy as np
import pytest
from moirais.fn.reldge import reliability_gebv


def test_reldge_basic():
    """Test basic functionality."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    result = reliability_gebv(fit)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_reldge_edge():
    """Test edge cases."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    result = reliability_gebv(fit)
    assert isinstance(result, dict)
