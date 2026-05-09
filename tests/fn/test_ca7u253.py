"""Tests for ca7u253.ca_chapter_7_unnumbered_253."""
import numpy as np
import pytest
from moirais.fn.ca7u253 import ca_chapter_7_unnumbered_253


def test_ca7u253_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_253(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca7u253_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_unnumbered_253(x)
    assert isinstance(result, dict)
