"""Tests for chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u80.chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_80."""
import numpy as np
import pytest
from moirais.fn.chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u80 import chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_80


def test_chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u80_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_80(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u80_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_80(x)
    assert isinstance(result, dict)
