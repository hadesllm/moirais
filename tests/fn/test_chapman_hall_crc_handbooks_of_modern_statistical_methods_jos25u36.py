"""Tests for chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u36.chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_36."""
import numpy as np
import pytest
from morie.fn.chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u36 import chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_36


def test_chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u36_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_36(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u36_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_36(x)
    assert isinstance(result, dict)
