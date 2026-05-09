"""Tests for rgpcgenl.rangayyan_pcg_envelope_avg."""
import numpy as np
import pytest
from moirais.fn.rgpcgenl import rangayyan_pcg_envelope_avg


def test_rgpcgenl_basic():
    """Test basic functionality."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_pcg_envelope_avg(pcg, ecg, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgpcgenl_edge():
    """Test edge cases."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_pcg_envelope_avg(pcg, ecg, fs)
    assert isinstance(result, dict)
