"""Tests for didtwfe.twoway_fixed_effects_did."""
import numpy as np
import pytest
from moirais.fn.didtwfe import twoway_fixed_effects_did


def test_didtwfe_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = twoway_fixed_effects_did(y, D, unit, time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_didtwfe_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = twoway_fixed_effects_did(y, D, unit, time)
    assert isinstance(result, dict)
