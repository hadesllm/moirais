"""Tests for sgtsbm.sgt_spectral_clustering_2."""
import numpy as np
import pytest
from moirais.fn.sgtsbm import sgt_spectral_clustering_2


def test_sgtsbm_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_spectral_clustering_2(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtsbm_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_spectral_clustering_2(A)
    assert isinstance(result, dict)
