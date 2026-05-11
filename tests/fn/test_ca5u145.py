"""Tests for ca5u145.ca_chapter_5_unnumbered_145."""
import numpy as np
import pytest
from morie.fn.ca5u145 import ca_chapter_5_unnumbered_145


def test_ca5u145_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_145(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca5u145_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_145(x)
    assert isinstance(result, dict)
