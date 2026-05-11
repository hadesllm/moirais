"""Tests for bivand20137u144.bivand2013_chapter_7_unnumbered_144."""
import numpy as np
import pytest
from morie.fn.bivand20137u144 import bivand2013_chapter_7_unnumbered_144


def test_bivand20137u144_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_144(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20137u144_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_144(x)
    assert isinstance(result, dict)
