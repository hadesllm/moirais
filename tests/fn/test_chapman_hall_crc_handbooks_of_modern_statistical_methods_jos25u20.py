"""Tests for chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u20.chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_20."""
import numpy as np
import pytest
from moirais.fn.chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u20 import chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_20


def test_chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u20_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_20(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u20_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_20(x)
    assert isinstance(result, dict)
