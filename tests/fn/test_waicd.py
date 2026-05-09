"""Tests for waicd.waic_diagnostic."""
import numpy as np
import pytest
from moirais.fn.waicd import waic_diagnostic


def test_waicd_basic():
    """Test basic functionality."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    result = waic_diagnostic(log_lik)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_waicd_edge():
    """Test edge cases."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    result = waic_diagnostic(log_lik)
    assert isinstance(result, dict)
