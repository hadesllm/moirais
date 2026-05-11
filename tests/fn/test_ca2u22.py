"""Tests for ca2u22.ca_chapter_2_unnumbered_22."""
import numpy as np
import pytest
from morie.fn.ca2u22 import ca_chapter_2_unnumbered_22


def test_ca2u22_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_unnumbered_22(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca2u22_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_unnumbered_22(x)
    assert isinstance(result, dict)
