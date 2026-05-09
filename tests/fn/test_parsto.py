"""Tests for parsto.parallel_trends_test."""
import numpy as np
import pytest
from moirais.fn.parsto import parallel_trends_test


def test_parsto_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    cohort = np.random.default_rng(42).normal(0, 1, 100)
    result = parallel_trends_test(y, D, unit, time, cohort)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_parsto_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    cohort = np.random.default_rng(42).normal(0, 1, 100)
    result = parallel_trends_test(y, D, unit, time, cohort)
    assert isinstance(result, dict)
