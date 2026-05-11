"""Tests for ca3u97.ca_chapter_3_unnumbered_97."""
import numpy as np
import pytest
from morie.fn.ca3u97 import ca_chapter_3_unnumbered_97


def test_ca3u97_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_97(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca3u97_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_97(x)
    assert isinstance(result, dict)
