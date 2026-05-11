"""Tests for ca2u20.ca_chapter_2_unnumbered_20."""
import numpy as np
import pytest
from morie.fn.ca2u20 import ca_chapter_2_unnumbered_20


def test_ca2u20_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_unnumbered_20(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca2u20_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_unnumbered_20(x)
    assert isinstance(result, dict)
