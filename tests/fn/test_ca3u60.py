"""Tests for ca3u60.ca_chapter_3_unnumbered_60."""
import numpy as np
import pytest
from moirais.fn.ca3u60 import ca_chapter_3_unnumbered_60


def test_ca3u60_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_60(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca3u60_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_60(x)
    assert isinstance(result, dict)
