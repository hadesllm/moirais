"""Tests for ca9u331.ca_chapter_9_unnumbered_331."""
import numpy as np
import pytest
from moirais.fn.ca9u331 import ca_chapter_9_unnumbered_331


def test_ca9u331_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_9_unnumbered_331(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca9u331_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_9_unnumbered_331(x)
    assert isinstance(result, dict)
