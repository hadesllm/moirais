"""Tests for ca5u147.ca_chapter_5_unnumbered_147."""
import numpy as np
import pytest
from moirais.fn.ca5u147 import ca_chapter_5_unnumbered_147


def test_ca5u147_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_147(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca5u147_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_147(x)
    assert isinstance(result, dict)
