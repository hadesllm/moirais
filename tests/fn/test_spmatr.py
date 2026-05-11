"""Tests for spmatr.schabenberger_matern_covariance."""
import numpy as np
import pytest
from morie.fn.spmatr import schabenberger_matern_covariance


def test_spmatr_basic():
    """Test basic functionality."""
    h = 0.3
    sigma2 = np.random.default_rng(42).normal(0, 1, 100)
    nu = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = schabenberger_matern_covariance(h, sigma2, nu, a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spmatr_edge():
    """Test edge cases."""
    h = 0.3
    sigma2 = np.random.default_rng(42).normal(0, 1, 100)
    nu = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = schabenberger_matern_covariance(h, sigma2, nu, a)
    assert isinstance(result, dict)
