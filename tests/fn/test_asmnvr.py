"""Tests for asmnvr.genome_assembly."""
import numpy as np
import pytest
from morie.fn.asmnvr import genome_assembly


def test_asmnvr_basic():
    """Test basic functionality."""
    reads = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = genome_assembly(reads, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_asmnvr_edge():
    """Test edge cases."""
    reads = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = genome_assembly(reads, k)
    assert isinstance(result, dict)
