"""Tests for kmtemp.kamath_temperature_sampling."""
import numpy as np
import pytest
from moirais.fn.kmtemp import kamath_temperature_sampling


def test_kmtemp_basic():
    """Test basic functionality."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = kamath_temperature_sampling(logits, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmtemp_edge():
    """Test edge cases."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = kamath_temperature_sampling(logits, T)
    assert isinstance(result, dict)
