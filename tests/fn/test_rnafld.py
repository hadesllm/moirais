"""Tests for rnafld.rna_fold."""
import numpy as np
import pytest
from morie.fn.rnafld import rna_fold


def test_rnafld_basic():
    """Test basic functionality."""
    sequence = np.random.default_rng(42).normal(0, 1, 100)
    result = rna_fold(sequence)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rnafld_edge():
    """Test edge cases."""
    sequence = np.random.default_rng(42).normal(0, 1, 100)
    result = rna_fold(sequence)
    assert isinstance(result, dict)
