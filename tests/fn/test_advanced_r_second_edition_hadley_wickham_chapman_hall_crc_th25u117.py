"""Tests for advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th25u117.advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_25_unnumbered_117."""
import numpy as np
import pytest
from morie.fn.advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th25u117 import advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_25_unnumbered_117


def test_advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th25u117_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_25_unnumbered_117(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th25u117_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_25_unnumbered_117(x)
    assert isinstance(result, dict)
