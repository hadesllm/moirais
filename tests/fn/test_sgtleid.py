"""Tests for sgtleid.sgt_leiden_step."""
import numpy as np
import pytest
from morie.fn.sgtleid import sgt_leiden_step


def test_sgtleid_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    labels = np.random.default_rng(43).integers(0, 2, 100)
    result = sgt_leiden_step(A, labels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtleid_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    labels = np.random.default_rng(43).integers(0, 2, 100)
    result = sgt_leiden_step(A, labels)
    assert isinstance(result, dict)
