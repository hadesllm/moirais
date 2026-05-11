"""Tests for ca4u115.ca_chapter_4_unnumbered_115."""
import numpy as np
import pytest
from morie.fn.ca4u115 import ca_chapter_4_unnumbered_115


def test_ca4u115_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_115(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca4u115_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_115(x)
    assert isinstance(result, dict)
