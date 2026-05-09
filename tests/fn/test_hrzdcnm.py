"""Tests for hrzdcnm.horowitz_deconv_normality."""
import numpy as np
import pytest
from moirais.fn.hrzdcnm import horowitz_deconv_normality


def test_hrzdcnm_basic():
    """Test basic functionality."""
    w = np.random.default_rng(45).exponential(1, 100)
    eps_density = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = horowitz_deconv_normality(w, eps_density, bandwidth, u)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzdcnm_edge():
    """Test edge cases."""
    w = np.random.default_rng(45).exponential(1, 100)
    eps_density = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = horowitz_deconv_normality(w, eps_density, bandwidth, u)
    assert isinstance(result, dict)
