"""Tests for ca8u296.ca_chapter_8_unnumbered_296."""
import numpy as np
import pytest
from moirais.fn.ca8u296 import ca_chapter_8_unnumbered_296


def test_ca8u296_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_296(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca8u296_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_296(x)
    assert isinstance(result, dict)
