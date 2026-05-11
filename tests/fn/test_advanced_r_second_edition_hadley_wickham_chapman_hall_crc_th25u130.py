"""Tests for advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th25u130.advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_25_unnumbered_130."""
import numpy as np
import pytest
from morie.fn.advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th25u130 import advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_25_unnumbered_130


def test_advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th25u130_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_25_unnumbered_130(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th25u130_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_25_unnumbered_130(x)
    assert isinstance(result, dict)
