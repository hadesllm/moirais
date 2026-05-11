"""Tests for ca3u65.ca_chapter_3_unnumbered_65."""
import numpy as np
import pytest
from morie.fn.ca3u65 import ca_chapter_3_unnumbered_65


def test_ca3u65_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_65(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca3u65_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_65(x)
    assert isinstance(result, dict)
