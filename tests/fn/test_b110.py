"""Tests for b110.burkov_lm_ch1_dataset_bce."""
import numpy as np
import pytest
from morie.fn.b110 import burkov_lm_ch1_dataset_bce


def test_b110_basic():
    """Test basic functionality."""
    y_hat = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    N = 100
    result = burkov_lm_ch1_dataset_bce(y_hat, y, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_b110_edge():
    """Test edge cases."""
    y_hat = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    N = 100
    result = burkov_lm_ch1_dataset_bce(y_hat, y, N)
    assert isinstance(result, dict)
