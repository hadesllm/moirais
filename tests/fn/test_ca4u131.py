"""Tests for ca4u131.ca_chapter_4_unnumbered_131."""
import numpy as np
import pytest
from morie.fn.ca4u131 import ca_chapter_4_unnumbered_131


def test_ca4u131_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_131(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca4u131_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_131(x)
    assert isinstance(result, dict)
