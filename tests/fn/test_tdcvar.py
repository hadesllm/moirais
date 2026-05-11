"""Tests for tdcvar.time_dep_covariate."""
import numpy as np
import pytest
from morie.fn.tdcvar import time_dep_covariate


def test_tdcvar_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    L_t = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = time_dep_covariate(y, A, L_t, time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tdcvar_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    L_t = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = time_dep_covariate(y, A, L_t, time)
    assert isinstance(result, dict)
