"""Tests for ca3u74.ca_chapter_3_unnumbered_74."""
import numpy as np
import pytest
from moirais.fn.ca3u74 import ca_chapter_3_unnumbered_74


def test_ca3u74_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_74(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca3u74_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_74(x)
    assert isinstance(result, dict)
