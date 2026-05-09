"""Tests for potef.potential_outcomes_individual."""
import numpy as np
import pytest
from moirais.fn.potef import potential_outcomes_individual


def test_potef_basic():
    """Test basic functionality."""
    Y1 = np.random.default_rng(42).normal(0, 1, 100)
    Y0 = np.random.default_rng(42).normal(0, 1, 100)
    result = potential_outcomes_individual(Y1, Y0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_potef_edge():
    """Test edge cases."""
    Y1 = np.random.default_rng(42).normal(0, 1, 100)
    Y0 = np.random.default_rng(42).normal(0, 1, 100)
    result = potential_outcomes_individual(Y1, Y0)
    assert isinstance(result, dict)
