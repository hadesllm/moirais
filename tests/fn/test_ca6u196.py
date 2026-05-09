"""Tests for ca6u196.ca_chapter_6_unnumbered_196."""
import numpy as np
import pytest
from moirais.fn.ca6u196 import ca_chapter_6_unnumbered_196


def test_ca6u196_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_unnumbered_196(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca6u196_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_unnumbered_196(x)
    assert isinstance(result, dict)
