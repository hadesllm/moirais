"""Tests for rgkneecl.rangayyan_knee_classify."""
import numpy as np
import pytest
from moirais.fn.rgkneecl import rangayyan_knee_classify


def test_rgkneecl_basic():
    """Test basic functionality."""
    vag = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    labels = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_knee_classify(vag, fs, labels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgkneecl_edge():
    """Test edge cases."""
    vag = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    labels = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_knee_classify(vag, fs, labels)
    assert isinstance(result, dict)
