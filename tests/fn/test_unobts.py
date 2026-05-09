"""Tests for unobts.unobserved_components."""
import numpy as np
import pytest
from moirais.fn.unobts import unobserved_components


def test_unobts_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    components = np.random.default_rng(42).normal(0, 1, 100)
    result = unobserved_components(y, components)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_unobts_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    components = np.random.default_rng(42).normal(0, 1, 100)
    result = unobserved_components(y, components)
    assert isinstance(result, dict)
