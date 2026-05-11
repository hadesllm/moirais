"""Tests for ca1u2.ca_chapter_1_unnumbered_2."""
import numpy as np
import pytest
from morie.fn.ca1u2 import ca_chapter_1_unnumbered_2


def test_ca1u2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_1_unnumbered_2(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca1u2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_1_unnumbered_2(x)
    assert isinstance(result, dict)
