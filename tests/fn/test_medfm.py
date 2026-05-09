"""Tests for medfm.mediation_formula."""
import numpy as np
import pytest
from moirais.fn.medfm import mediation_formula


def test_medfm_basic():
    """Test basic functionality."""
    Y_model = np.random.default_rng(42).normal(0, 1, 100)
    M_model = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    x_prime = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = mediation_formula(Y_model, M_model, x, x_prime, C)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_medfm_edge():
    """Test edge cases."""
    Y_model = np.random.default_rng(42).normal(0, 1, 100)
    M_model = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    x_prime = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = mediation_formula(Y_model, M_model, x, x_prime, C)
    assert isinstance(result, dict)
