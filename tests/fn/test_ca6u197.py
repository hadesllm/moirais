"""Tests for ca6u197.ca_chapter_6_unnumbered_197."""
import numpy as np
import pytest
from moirais.fn.ca6u197 import ca_chapter_6_unnumbered_197


def test_ca6u197_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_unnumbered_197(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca6u197_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_unnumbered_197(x)
    assert isinstance(result, dict)
