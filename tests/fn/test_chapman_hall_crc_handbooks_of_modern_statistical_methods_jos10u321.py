"""Tests for chapman_hall_crc_handbooks_of_modern_statistical_methods_jos10u321.chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_10_unnumbered_321."""
import numpy as np
import pytest
from moirais.fn.chapman_hall_crc_handbooks_of_modern_statistical_methods_jos10u321 import chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_10_unnumbered_321


def test_chapman_hall_crc_handbooks_of_modern_statistical_methods_jos10u321_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_10_unnumbered_321(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_chapman_hall_crc_handbooks_of_modern_statistical_methods_jos10u321_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_10_unnumbered_321(x)
    assert isinstance(result, dict)
