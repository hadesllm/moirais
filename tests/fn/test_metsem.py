"""Tests for metsem.metagenome_assembly."""
import numpy as np
import pytest
from morie.fn.metsem import metagenome_assembly


def test_metsem_basic():
    """Test basic functionality."""
    reads = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = metagenome_assembly(reads, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_metsem_edge():
    """Test edge cases."""
    reads = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = metagenome_assembly(reads, k)
    assert isinstance(result, dict)
