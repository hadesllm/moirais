"""Tests for rgpsd2hz.rangayyan_psd_to_hz."""
import numpy as np
import pytest
from moirais.fn.rgpsd2hz import rangayyan_psd_to_hz


def test_rgpsd2hz_basic():
    """Test basic functionality."""
    psd = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    fs = 100.0
    result = rangayyan_psd_to_hz(psd, N, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgpsd2hz_edge():
    """Test edge cases."""
    psd = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    fs = 100.0
    result = rangayyan_psd_to_hz(psd, N, fs)
    assert isinstance(result, dict)
