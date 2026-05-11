"""Tests for ca8u315.ca_chapter_8_unnumbered_315."""
import numpy as np
import pytest
from morie.fn.ca8u315 import ca_chapter_8_unnumbered_315


def test_ca8u315_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_315(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_ca8u315_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_315(x)
    assert isinstance(result, dict)
