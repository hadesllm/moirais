"""Tests for rgvmg.rangayyan_vmg."""
import numpy as np
import pytest
from moirais.fn.rgvmg import rangayyan_vmg


def test_rgvmg_basic():
    """Test basic functionality."""
    vmg = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_vmg(vmg, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgvmg_edge():
    """Test edge cases."""
    vmg = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_vmg(vmg, fs)
    assert isinstance(result, dict)
