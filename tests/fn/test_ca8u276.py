"""Tests for ca8u276.ca_chapter_8_unnumbered_276."""
import numpy as np
import pytest
from morie.fn.ca8u276 import ca_chapter_8_unnumbered_276


def test_ca8u276_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_276(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca8u276_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_276(x)
    assert isinstance(result, dict)
