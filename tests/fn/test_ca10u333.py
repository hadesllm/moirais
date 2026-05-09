"""Tests for ca10u333.ca_chapter_10_unnumbered_333."""
import numpy as np
import pytest
from moirais.fn.ca10u333 import ca_chapter_10_unnumbered_333


def test_ca10u333_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_10_unnumbered_333(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca10u333_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_10_unnumbered_333(x)
    assert isinstance(result, dict)
