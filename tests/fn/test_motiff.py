"""Tests for motiff.motif_count."""
import numpy as np
import pytest
from moirais.fn.motiff import motif_count


def test_motiff_basic():
    """Test basic functionality."""
    G = np.eye(10)
    motif_size = 100
    result = motif_count(G, motif_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_motiff_edge():
    """Test edge cases."""
    G = np.eye(10)
    motif_size = 100
    result = motif_count(G, motif_size)
    assert isinstance(result, dict)
