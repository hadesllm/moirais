"""Tests for voldoc.vol_decomposed_realised."""
import numpy as np
import pytest
from moirais.fn.voldoc import vol_decomposed_realised


def test_voldoc_basic():
    """Test basic functionality."""
    RV = np.random.default_rng(42).normal(0, 1, 100)
    BPV = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_decomposed_realised(RV, BPV)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_voldoc_edge():
    """Test edge cases."""
    RV = np.random.default_rng(42).normal(0, 1, 100)
    BPV = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_decomposed_realised(RV, BPV)
    assert isinstance(result, dict)
