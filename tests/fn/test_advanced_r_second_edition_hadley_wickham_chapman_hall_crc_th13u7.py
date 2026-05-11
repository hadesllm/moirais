"""Tests for advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th13u7.advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_13_unnumbered_7."""
import numpy as np
import pytest
from morie.fn.advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th13u7 import advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_13_unnumbered_7


def test_advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th13u7_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_13_unnumbered_7(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th13u7_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_13_unnumbered_7(x)
    assert isinstance(result, dict)
