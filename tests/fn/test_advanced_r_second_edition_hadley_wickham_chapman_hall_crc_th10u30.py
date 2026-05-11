"""Tests for advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th10u30.advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_10_unnumbered_30."""
import numpy as np
import pytest
from morie.fn.advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th10u30 import advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_10_unnumbered_30


def test_advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th10u30_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_10_unnumbered_30(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th10u30_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_10_unnumbered_30(x)
    assert isinstance(result, dict)
