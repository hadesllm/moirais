"""Tests for rdrobu.calonico_cattaneo_titiunik."""
import numpy as np
import pytest
from moirais.fn.rdrobu import calonico_cattaneo_titiunik


def test_rdrobu_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    result = calonico_cattaneo_titiunik(y, x, cutoff)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rdrobu_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    result = calonico_cattaneo_titiunik(y, x, cutoff)
    assert isinstance(result, dict)
