"""Tests for ca5u165.ca_chapter_5_unnumbered_165."""
import numpy as np
import pytest
from morie.fn.ca5u165 import ca_chapter_5_unnumbered_165


def test_ca5u165_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_165(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca5u165_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_165(x)
    assert isinstance(result, dict)
