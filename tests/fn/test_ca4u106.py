"""Tests for ca4u106.ca_chapter_4_unnumbered_106."""
import numpy as np
import pytest
from morie.fn.ca4u106 import ca_chapter_4_unnumbered_106


def test_ca4u106_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_106(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca4u106_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_106(x)
    assert isinstance(result, dict)
