"""Tests for motfsr.motif_meme."""
import numpy as np
import pytest
from morie.fn.motfsr import motif_meme


def test_motfsr_basic():
    """Test basic functionality."""
    sequences = np.random.default_rng(42).normal(0, 1, 100)
    motif_length = np.random.default_rng(42).normal(0, 1, 100)
    result = motif_meme(sequences, motif_length)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_motfsr_edge():
    """Test edge cases."""
    sequences = np.random.default_rng(42).normal(0, 1, 100)
    motif_length = np.random.default_rng(42).normal(0, 1, 100)
    result = motif_meme(sequences, motif_length)
    assert isinstance(result, dict)
