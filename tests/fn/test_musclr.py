"""Tests for musclr.muscle_msa."""
import numpy as np
import pytest
from moirais.fn.musclr import muscle_msa


def test_musclr_basic():
    """Test basic functionality."""
    sequences = np.random.default_rng(42).normal(0, 1, 100)
    result = muscle_msa(sequences)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_musclr_edge():
    """Test edge cases."""
    sequences = np.random.default_rng(42).normal(0, 1, 100)
    result = muscle_msa(sequences)
    assert isinstance(result, dict)
