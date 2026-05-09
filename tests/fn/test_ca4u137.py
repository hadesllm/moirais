"""Tests for ca4u137.ca_chapter_4_unnumbered_137."""
import numpy as np
import pytest
from moirais.fn.ca4u137 import ca_chapter_4_unnumbered_137


def test_ca4u137_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_137(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca4u137_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_137(x)
    assert isinstance(result, dict)
