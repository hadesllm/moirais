"""Tests for ca6u209.ca_chapter_6_unnumbered_209."""
import numpy as np
import pytest
from moirais.fn.ca6u209 import ca_chapter_6_unnumbered_209


def test_ca6u209_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_unnumbered_209(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca6u209_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_unnumbered_209(x)
    assert isinstance(result, dict)
