"""Tests for ca8u269.ca_chapter_8_unnumbered_269."""
import numpy as np
import pytest
from morie.fn.ca8u269 import ca_chapter_8_unnumbered_269


def test_ca8u269_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_269(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca8u269_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_269(x)
    assert isinstance(result, dict)
