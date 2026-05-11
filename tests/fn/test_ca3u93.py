"""Tests for ca3u93.ca_chapter_3_unnumbered_93."""
import numpy as np
import pytest
from morie.fn.ca3u93 import ca_chapter_3_unnumbered_93


def test_ca3u93_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_93(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca3u93_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_93(x)
    assert isinstance(result, dict)
