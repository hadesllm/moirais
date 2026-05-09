"""Tests for km105.kamath_ch6_gedi_combined_loss."""
import numpy as np
import pytest
from moirais.fn.km105 import kamath_ch6_gedi_combined_loss


def test_km105_basic():
    """Test basic functionality."""
    L_g = np.random.default_rng(42).normal(0, 1, 100)
    L_d = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = kamath_ch6_gedi_combined_loss(L_g, L_d, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km105_edge():
    """Test edge cases."""
    L_g = np.random.default_rng(42).normal(0, 1, 100)
    L_d = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = kamath_ch6_gedi_combined_loss(L_g, L_d, lam)
    assert isinstance(result, dict)
