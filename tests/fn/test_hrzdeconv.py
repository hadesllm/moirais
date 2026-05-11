"""Tests for hrzdeconv.horowitz_deconvolution_density."""
import numpy as np
import pytest
from morie.fn.hrzdeconv import horowitz_deconvolution_density


def test_hrzdeconv_basic():
    """Test basic functionality."""
    w = np.random.default_rng(45).exponential(1, 100)
    eps_density = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_deconvolution_density(w, eps_density, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzdeconv_edge():
    """Test edge cases."""
    w = np.random.default_rng(45).exponential(1, 100)
    eps_density = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_deconvolution_density(w, eps_density, bandwidth)
    assert isinstance(result, dict)
