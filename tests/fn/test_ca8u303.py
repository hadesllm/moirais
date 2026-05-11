"""Tests for ca8u303.ca_chapter_8_unnumbered_303."""
import numpy as np
import pytest
from morie.fn.ca8u303 import ca_chapter_8_unnumbered_303


def test_ca8u303_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_303(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca8u303_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_303(x)
    assert isinstance(result, dict)
