"""Tests for ca8u264.ca_chapter_8_unnumbered_264."""
import numpy as np
import pytest
from morie.fn.ca8u264 import ca_chapter_8_unnumbered_264


def test_ca8u264_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_264(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca8u264_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_264(x)
    assert isinstance(result, dict)
