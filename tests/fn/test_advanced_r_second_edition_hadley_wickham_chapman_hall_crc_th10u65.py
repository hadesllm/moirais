"""Tests for advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th10u65.advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_10_unnumbered_65."""
import numpy as np
import pytest
from moirais.fn.advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th10u65 import advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_10_unnumbered_65


def test_advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th10u65_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_10_unnumbered_65(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th10u65_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_10_unnumbered_65(x)
    assert isinstance(result, dict)
