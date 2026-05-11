"""Tests for ca12u361.ca_chapter_12_unnumbered_361."""
import numpy as np
import pytest
from morie.fn.ca12u361 import ca_chapter_12_unnumbered_361


def test_ca12u361_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_12_unnumbered_361(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca12u361_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_12_unnumbered_361(x)
    assert isinstance(result, dict)
