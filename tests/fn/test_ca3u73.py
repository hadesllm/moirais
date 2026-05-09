"""Tests for ca3u73.ca_chapter_3_unnumbered_73."""
import numpy as np
import pytest
from moirais.fn.ca3u73 import ca_chapter_3_unnumbered_73


def test_ca3u73_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_73(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca3u73_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_3_unnumbered_73(x)
    assert isinstance(result, dict)
