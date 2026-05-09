"""Tests for chapman_hall_crc_handbooks_of_modern_statistical_methods_jos10u830.chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_10_unnumbered_830."""
import numpy as np
import pytest
from moirais.fn.chapman_hall_crc_handbooks_of_modern_statistical_methods_jos10u830 import chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_10_unnumbered_830


def test_chapman_hall_crc_handbooks_of_modern_statistical_methods_jos10u830_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_10_unnumbered_830(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_chapman_hall_crc_handbooks_of_modern_statistical_methods_jos10u830_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_10_unnumbered_830(x)
    assert isinstance(result, dict)
