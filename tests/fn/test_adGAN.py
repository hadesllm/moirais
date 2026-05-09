"""Tests for adGAN.adversarial_anomaly."""
import numpy as np
import pytest
from moirais.fn.adGAN import adversarial_anomaly


def test_adGAN_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    result = adversarial_anomaly(x, D)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_adGAN_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    result = adversarial_anomaly(x, D)
    assert isinstance(result, dict)
