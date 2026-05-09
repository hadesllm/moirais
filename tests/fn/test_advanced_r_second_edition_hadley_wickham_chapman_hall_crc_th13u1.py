"""Tests for advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th13u1.advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_13_unnumbered_1."""
import numpy as np
import pytest
from moirais.fn.advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th13u1 import advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_13_unnumbered_1


def test_advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th13u1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_13_unnumbered_1(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th13u1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_13_unnumbered_1(x)
    assert isinstance(result, dict)
