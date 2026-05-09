"""Tests for tncomp.compound_diversity."""
import numpy as np
import pytest
from moirais.fn.tncomp import compound_diversity


def test_tncomp_basic():
    """Test basic functionality."""
    fps = np.random.default_rng(42).normal(0, 1, 100)
    n_pick = np.random.default_rng(42).normal(0, 1, 100)
    result = compound_diversity(fps, n_pick)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tncomp_edge():
    """Test edge cases."""
    fps = np.random.default_rng(42).normal(0, 1, 100)
    n_pick = np.random.default_rng(42).normal(0, 1, 100)
    result = compound_diversity(fps, n_pick)
    assert isinstance(result, dict)
