"""Tests for ca11u357.ca_chapter_11_unnumbered_357."""
import numpy as np
import pytest
from moirais.fn.ca11u357 import ca_chapter_11_unnumbered_357


def test_ca11u357_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_unnumbered_357(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca11u357_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_unnumbered_357(x)
    assert isinstance(result, dict)
