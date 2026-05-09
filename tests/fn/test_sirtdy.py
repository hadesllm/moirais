"""Tests for sirtdy.sir_age_structured."""
import numpy as np
import pytest
from moirais.fn.sirtdy import sir_age_structured


def test_sirtdy_basic():
    """Test basic functionality."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    I = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    contact_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    gamma = 1.0
    result = sir_age_structured(S, I, R, contact_matrix, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sirtdy_edge():
    """Test edge cases."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    I = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    contact_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    gamma = 1.0
    result = sir_age_structured(S, I, R, contact_matrix, gamma)
    assert isinstance(result, dict)
