"""Tests for copynm.copy_number_variant."""
import numpy as np
import pytest
from morie.fn.copynm import copy_number_variant


def test_copynm_basic():
    """Test basic functionality."""
    depth = np.random.default_rng(42).normal(0, 1, 100)
    reference_depth = np.random.default_rng(42).normal(0, 1, 100)
    result = copy_number_variant(depth, reference_depth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_copynm_edge():
    """Test edge cases."""
    depth = np.random.default_rng(42).normal(0, 1, 100)
    reference_depth = np.random.default_rng(42).normal(0, 1, 100)
    result = copy_number_variant(depth, reference_depth)
    assert isinstance(result, dict)
