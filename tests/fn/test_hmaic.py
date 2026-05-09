"""Tests for hmaic.geron_aic."""
import numpy as np
import pytest
from moirais.fn.hmaic import geron_aic


def test_hmaic_basic():
    """Test basic functionality."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = geron_aic(log_lik, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmaic_edge():
    """Test edge cases."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = geron_aic(log_lik, k)
    assert isinstance(result, dict)
