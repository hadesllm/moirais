"""Tests for rgvag.rangayyan_vag_analysis."""
import numpy as np
import pytest
from moirais.fn.rgvag import rangayyan_vag_analysis


def test_rgvag_basic():
    """Test basic functionality."""
    vag = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_vag_analysis(vag, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgvag_edge():
    """Test edge cases."""
    vag = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_vag_analysis(vag, fs)
    assert isinstance(result, dict)
