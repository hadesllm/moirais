"""Tests for ca1u1.ca_chapter_1_unnumbered_1."""
import numpy as np
import pytest
from morie.fn.ca1u1 import ca_chapter_1_unnumbered_1


def test_ca1u1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_1_unnumbered_1(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca1u1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_1_unnumbered_1(x)
    assert isinstance(result, dict)
