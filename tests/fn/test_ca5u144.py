"""Tests for ca5u144.ca_chapter_5_unnumbered_144."""
import numpy as np
import pytest
from morie.fn.ca5u144 import ca_chapter_5_unnumbered_144


def test_ca5u144_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_144(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca5u144_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_144(x)
    assert isinstance(result, dict)
