"""Tests for ca1u14.ca_chapter_1_unnumbered_14."""
import numpy as np
import pytest
from morie.fn.ca1u14 import ca_chapter_1_unnumbered_14


def test_ca1u14_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_1_unnumbered_14(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca1u14_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_1_unnumbered_14(x)
    assert isinstance(result, dict)
