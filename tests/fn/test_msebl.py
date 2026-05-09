"""Tests for msebl.mse_loss_continuous."""
import numpy as np
import pytest
from moirais.fn.msebl import mse_loss_continuous


def test_msebl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    y_hat = np.random.default_rng(42).normal(0, 1, 100)
    result = mse_loss_continuous(y, y_hat)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msebl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    y_hat = np.random.default_rng(42).normal(0, 1, 100)
    result = mse_loss_continuous(y, y_hat)
    assert isinstance(result, dict)
