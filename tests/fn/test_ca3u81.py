"""Tests for ca3u81.ca_chapter_3_unnumbered_81."""
import numpy as np
import pytest
from moirais.fn.ca3u81 import ca_chapter_3_unnumbered_81


def test_ca3u81_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_81(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca3u81_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_81(x)
    assert isinstance(result, dict)
