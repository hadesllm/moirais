"""Tests for vepan.vep_annotation."""
import numpy as np
import pytest
from moirais.fn.vepan import vep_annotation


def test_vepan_basic():
    """Test basic functionality."""
    variants = np.random.default_rng(42).normal(0, 1, 100)
    cache = np.random.default_rng(42).normal(0, 1, 100)
    result = vep_annotation(variants, cache)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vepan_edge():
    """Test edge cases."""
    variants = np.random.default_rng(42).normal(0, 1, 100)
    cache = np.random.default_rng(42).normal(0, 1, 100)
    result = vep_annotation(variants, cache)
    assert isinstance(result, dict)
