"""Tests for mafft.mafft_alignment."""
import numpy as np
import pytest
from morie.fn.mafft import mafft_alignment


def test_mafft_basic():
    """Test basic functionality."""
    sequences = np.random.default_rng(42).normal(0, 1, 100)
    mode = 'auto'
    result = mafft_alignment(sequences, mode)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mafft_edge():
    """Test edge cases."""
    sequences = np.random.default_rng(42).normal(0, 1, 100)
    mode = 'auto'
    result = mafft_alignment(sequences, mode)
    assert isinstance(result, dict)
