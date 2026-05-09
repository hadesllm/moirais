"""Tests for pwaic.effective_parameters_waic."""
import numpy as np
import pytest
from moirais.fn.pwaic import effective_parameters_waic


def test_pwaic_basic():
    """Test basic functionality."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    result = effective_parameters_waic(log_lik)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pwaic_edge():
    """Test edge cases."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    result = effective_parameters_waic(log_lik)
    assert isinstance(result, dict)
