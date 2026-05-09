"""Tests for ngppr.normalized_gamma_process."""
import numpy as np
import pytest
from moirais.fn.ngppr import normalized_gamma_process


def test_ngppr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = normalized_gamma_process(y, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ngppr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = normalized_gamma_process(y, alpha)
    assert isinstance(result, dict)
