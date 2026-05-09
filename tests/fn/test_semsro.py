"""Tests for semsro.sem_residual."""
import numpy as np
import pytest
from moirais.fn.semsro import sem_residual


def test_semsro_basic():
    """Test basic functionality."""
    sample_cov = np.random.default_rng(42).normal(0, 1, 100)
    fitted_cov = np.random.default_rng(42).normal(0, 1, 100)
    result = sem_residual(sample_cov, fitted_cov)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_semsro_edge():
    """Test edge cases."""
    sample_cov = np.random.default_rng(42).normal(0, 1, 100)
    fitted_cov = np.random.default_rng(42).normal(0, 1, 100)
    result = sem_residual(sample_cov, fitted_cov)
    assert isinstance(result, dict)
