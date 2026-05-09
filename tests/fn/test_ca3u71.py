"""Tests for ca3u71.ca_chapter_3_unnumbered_71."""
import numpy as np
import pytest
from moirais.fn.ca3u71 import ca_chapter_3_unnumbered_71


def test_ca3u71_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_71(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca3u71_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_71(x)
    assert isinstance(result, dict)
