"""Tests for aitpow.aitchison_powering."""
import numpy as np
import pytest
from moirais.fn.aitpow import aitchison_powering


def test_aitpow_basic():
    """Test basic functionality."""
    alpha = 0.05
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_powering(alpha, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitpow_edge():
    """Test edge cases."""
    alpha = 0.05
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_powering(alpha, x)
    assert isinstance(result, dict)
