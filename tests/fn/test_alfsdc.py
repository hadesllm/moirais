"""Tests for alfsdc.alphafold_sidechain."""
import numpy as np
import pytest
from moirais.fn.alfsdc import alphafold_sidechain


def test_alfsdc_basic():
    """Test basic functionality."""
    s = 90
    frames = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_sidechain(s, frames)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfsdc_edge():
    """Test edge cases."""
    s = 90
    frames = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_sidechain(s, frames)
    assert isinstance(result, dict)
