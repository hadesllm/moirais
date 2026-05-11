"""Tests for protdis.protein_disorder."""
import numpy as np
import pytest
from morie.fn.protdis import protein_disorder


def test_protdis_basic():
    """Test basic functionality."""
    sequence = np.random.default_rng(42).normal(0, 1, 100)
    result = protein_disorder(sequence)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_protdis_edge():
    """Test edge cases."""
    sequence = np.random.default_rng(42).normal(0, 1, 100)
    result = protein_disorder(sequence)
    assert isinstance(result, dict)
