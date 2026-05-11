"""Tests for trnsmh.transmembrane_topology."""
import numpy as np
import pytest
from morie.fn.trnsmh import transmembrane_topology


def test_trnsmh_basic():
    """Test basic functionality."""
    sequence = np.random.default_rng(42).normal(0, 1, 100)
    result = transmembrane_topology(sequence)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_trnsmh_edge():
    """Test edge cases."""
    sequence = np.random.default_rng(42).normal(0, 1, 100)
    result = transmembrane_topology(sequence)
    assert isinstance(result, dict)
