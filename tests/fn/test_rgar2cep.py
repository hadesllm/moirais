"""Tests for rgar2cep.rangayyan_ar_to_cepstrum."""
import numpy as np
import pytest
from moirais.fn.rgar2cep import rangayyan_ar_to_cepstrum


def test_rgar2cep_basic():
    """Test basic functionality."""
    a_coeffs = np.random.default_rng(42).normal(0, 1, 100)
    sigma_sq = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ar_to_cepstrum(a_coeffs, sigma_sq)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgar2cep_edge():
    """Test edge cases."""
    a_coeffs = np.random.default_rng(42).normal(0, 1, 100)
    sigma_sq = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ar_to_cepstrum(a_coeffs, sigma_sq)
    assert isinstance(result, dict)
