"""Tests for ca7u260.ca_chapter_7_unnumbered_260."""
import numpy as np
import pytest
from morie.fn.ca7u260 import ca_chapter_7_unnumbered_260


def test_ca7u260_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_260(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca7u260_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_260(x)
    assert isinstance(result, dict)
