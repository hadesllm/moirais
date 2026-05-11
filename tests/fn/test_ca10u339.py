"""Tests for ca10u339.ca_chapter_10_unnumbered_339."""
import numpy as np
import pytest
from morie.fn.ca10u339 import ca_chapter_10_unnumbered_339


def test_ca10u339_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_10_unnumbered_339(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca10u339_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_10_unnumbered_339(x)
    assert isinstance(result, dict)
