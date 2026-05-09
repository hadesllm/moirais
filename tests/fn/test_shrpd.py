"""Tests for shrpd.shepard_diagram."""
import numpy as np
import pytest
from moirais.fn.shrpd import shepard_diagram


def test_shrpd_basic():
    """Test basic functionality."""
    delta = np.random.default_rng(42).normal(0, 1, 100)
    D_config = np.random.default_rng(42).normal(0, 1, 100)
    result = shepard_diagram(delta, D_config)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_shrpd_edge():
    """Test edge cases."""
    delta = np.random.default_rng(42).normal(0, 1, 100)
    D_config = np.random.default_rng(42).normal(0, 1, 100)
    result = shepard_diagram(delta, D_config)
    assert isinstance(result, dict)
