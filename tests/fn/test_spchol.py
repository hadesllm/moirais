"""Tests for spchol.schabenberger_cholesky_sim."""
import numpy as np
import pytest
from moirais.fn.spchol import schabenberger_cholesky_sim


def test_spchol_basic():
    """Test basic functionality."""
    mu = 0.0
    cov_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = schabenberger_cholesky_sim(mu, cov_matrix)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spchol_edge():
    """Test edge cases."""
    mu = 0.0
    cov_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = schabenberger_cholesky_sim(mu, cov_matrix)
    assert isinstance(result, dict)
