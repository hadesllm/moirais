"""Tests for clusmd.butina_cluster."""
import numpy as np
import pytest
from moirais.fn.clusmd import butina_cluster


def test_clusmd_basic():
    """Test basic functionality."""
    fps = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    result = butina_cluster(fps, cutoff)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_clusmd_edge():
    """Test edge cases."""
    fps = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    result = butina_cluster(fps, cutoff)
    assert isinstance(result, dict)
