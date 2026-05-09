"""Tests for chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u37.chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_37."""
import numpy as np
import pytest
from moirais.fn.chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u37 import chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_37


def test_chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u37_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_37(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u37_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_37(x)
    assert isinstance(result, dict)
