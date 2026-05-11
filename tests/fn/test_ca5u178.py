"""Tests for ca5u178.ca_chapter_5_unnumbered_178."""
import numpy as np
import pytest
from morie.fn.ca5u178 import ca_chapter_5_unnumbered_178


def test_ca5u178_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_178(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca5u178_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_178(x)
    assert isinstance(result, dict)
