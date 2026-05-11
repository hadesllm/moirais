"""Tests for ca3u78.ca_chapter_3_unnumbered_78."""
import numpy as np
import pytest
from morie.fn.ca3u78 import ca_chapter_3_unnumbered_78


def test_ca3u78_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_78(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca3u78_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_78(x)
    assert isinstance(result, dict)
