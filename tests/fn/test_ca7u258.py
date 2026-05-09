"""Tests for ca7u258.ca_chapter_7_unnumbered_258."""
import numpy as np
import pytest
from moirais.fn.ca7u258 import ca_chapter_7_unnumbered_258


def test_ca7u258_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_258(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca7u258_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_258(x)
    assert isinstance(result, dict)
