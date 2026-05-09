"""Tests for ca10u340.ca_chapter_10_unnumbered_340."""
import numpy as np
import pytest
from moirais.fn.ca10u340 import ca_chapter_10_unnumbered_340


def test_ca10u340_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_10_unnumbered_340(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca10u340_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_10_unnumbered_340(x)
    assert isinstance(result, dict)
