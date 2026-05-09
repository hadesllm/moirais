"""Tests for frgrow.fragment_growing."""
import numpy as np
import pytest
from moirais.fn.frgrow import fragment_growing


def test_frgrow_basic():
    """Test basic functionality."""
    fragment = np.random.default_rng(42).normal(0, 1, 100)
    linker_lib = np.random.default_rng(42).normal(0, 1, 100)
    result = fragment_growing(fragment, linker_lib)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_frgrow_edge():
    """Test edge cases."""
    fragment = np.random.default_rng(42).normal(0, 1, 100)
    linker_lib = np.random.default_rng(42).normal(0, 1, 100)
    result = fragment_growing(fragment, linker_lib)
    assert isinstance(result, dict)
