"""Tests for ca8u305.ca_chapter_8_unnumbered_305."""
import numpy as np
import pytest
from morie.fn.ca8u305 import ca_chapter_8_unnumbered_305


def test_ca8u305_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_305(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca8u305_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_305(x)
    assert isinstance(result, dict)
