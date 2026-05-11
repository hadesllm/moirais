"""Tests for rng255.rangayyan_ch4_power_cepstrum_relation_to_complex."""
import numpy as np
import pytest
from morie.fn.rng255 import rangayyan_ch4_power_cepstrum_relation_to_complex


def test_rng255_basic():
    """Test basic functionality."""
    y_hat = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_power_cepstrum_relation_to_complex(y_hat, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng255_edge():
    """Test edge cases."""
    y_hat = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_power_cepstrum_relation_to_complex(y_hat, n)
    assert isinstance(result, dict)
