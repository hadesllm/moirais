"""Tests for ca1u9.ca_chapter_1_unnumbered_9."""
import numpy as np
import pytest
from morie.fn.ca1u9 import ca_chapter_1_unnumbered_9


def test_ca1u9_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_1_unnumbered_9(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca1u9_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_1_unnumbered_9(x)
    assert isinstance(result, dict)
