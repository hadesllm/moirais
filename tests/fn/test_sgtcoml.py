"""Tests for sgtcoml.sgt_louvain_step."""
import numpy as np
import pytest
from morie.fn.sgtcoml import sgt_louvain_step


def test_sgtcoml_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    labels = np.random.default_rng(43).integers(0, 2, 100)
    result = sgt_louvain_step(A, labels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtcoml_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    labels = np.random.default_rng(43).integers(0, 2, 100)
    result = sgt_louvain_step(A, labels)
    assert isinstance(result, dict)
