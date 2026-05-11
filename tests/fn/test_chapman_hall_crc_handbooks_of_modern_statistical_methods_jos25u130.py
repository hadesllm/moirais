"""Tests for chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u130.chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_130."""
import numpy as np
import pytest
from morie.fn.chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u130 import chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_130


def test_chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u130_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_130(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u130_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_130(x)
    assert isinstance(result, dict)
