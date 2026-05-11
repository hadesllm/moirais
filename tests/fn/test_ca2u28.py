"""Tests for ca2u28.ca_chapter_2_unnumbered_28."""
import numpy as np
import pytest
from morie.fn.ca2u28 import ca_chapter_2_unnumbered_28


def test_ca2u28_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_unnumbered_28(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca2u28_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_unnumbered_28(x)
    assert isinstance(result, dict)
