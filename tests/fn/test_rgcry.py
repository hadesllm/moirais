"""Tests for rgcry.rangayyan_infant_cry."""
import numpy as np
import pytest
from moirais.fn.rgcry import rangayyan_infant_cry


def test_rgcry_basic():
    """Test basic functionality."""
    cry = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_infant_cry(cry, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgcry_edge():
    """Test edge cases."""
    cry = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_infant_cry(cry, fs)
    assert isinstance(result, dict)
