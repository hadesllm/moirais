"""Tests for ca8u272.ca_chapter_8_unnumbered_272."""
import numpy as np
import pytest
from morie.fn.ca8u272 import ca_chapter_8_unnumbered_272


def test_ca8u272_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_272(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca8u272_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_272(x)
    assert isinstance(result, dict)
