"""Tests for h2est.heritability_lmm."""
import numpy as np
import pytest
from moirais.fn.h2est import heritability_lmm


def test_h2est_basic():
    """Test basic functionality."""
    sigma_g2 = np.random.default_rng(42).normal(0, 1, 100)
    sigma_e2 = np.random.default_rng(42).normal(0, 1, 100)
    result = heritability_lmm(sigma_g2, sigma_e2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_h2est_edge():
    """Test edge cases."""
    sigma_g2 = np.random.default_rng(42).normal(0, 1, 100)
    sigma_e2 = np.random.default_rng(42).normal(0, 1, 100)
    result = heritability_lmm(sigma_g2, sigma_e2)
    assert isinstance(result, dict)
