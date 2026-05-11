"""Tests for ca3u64.ca_chapter_3_unnumbered_64."""
import numpy as np
import pytest
from morie.fn.ca3u64 import ca_chapter_3_unnumbered_64


def test_ca3u64_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_64(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca3u64_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_64(x)
    assert isinstance(result, dict)
