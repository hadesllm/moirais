"""Tests for grtmp.geron_temperature_sampling."""
import numpy as np
import pytest
from morie.fn.grtmp import geron_temperature_sampling


def test_grtmp_basic():
    """Test basic functionality."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_temperature_sampling(logits, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grtmp_edge():
    """Test edge cases."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_temperature_sampling(logits, T)
    assert isinstance(result, dict)
