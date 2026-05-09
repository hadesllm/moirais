"""Tests for alfcrp.alphafold_cropping."""
import numpy as np
import pytest
from moirais.fn.alfcrp import alphafold_cropping


def test_alfcrp_basic():
    """Test basic functionality."""
    sequence = np.random.default_rng(42).normal(0, 1, 100)
    crop_size = 100
    result = alphafold_cropping(sequence, crop_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfcrp_edge():
    """Test edge cases."""
    sequence = np.random.default_rng(42).normal(0, 1, 100)
    crop_size = 100
    result = alphafold_cropping(sequence, crop_size)
    assert isinstance(result, dict)
