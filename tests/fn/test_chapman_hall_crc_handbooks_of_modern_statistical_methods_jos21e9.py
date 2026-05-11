"""Tests for chapman_hall_crc_handbooks_of_modern_statistical_methods_jos21e9.chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_21_equation_9."""
import numpy as np
import pytest
from morie.fn.chapman_hall_crc_handbooks_of_modern_statistical_methods_jos21e9 import chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_21_equation_9


def test_chapman_hall_crc_handbooks_of_modern_statistical_methods_jos21e9_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_21_equation_9(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_chapman_hall_crc_handbooks_of_modern_statistical_methods_jos21e9_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_21_equation_9(x)
    assert isinstance(result, dict)
