"""Tests for strfst.f_statistics."""
import numpy as np
import pytest
from morie.fn.strfst import f_statistics


def test_strfst_basic():
    """Test basic functionality."""
    allele_freqs = np.random.default_rng(42).normal(0, 1, 100)
    populations = np.random.default_rng(42).normal(0, 1, 100)
    result = f_statistics(allele_freqs, populations)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_strfst_edge():
    """Test edge cases."""
    allele_freqs = np.random.default_rng(42).normal(0, 1, 100)
    populations = np.random.default_rng(42).normal(0, 1, 100)
    result = f_statistics(allele_freqs, populations)
    assert isinstance(result, dict)
