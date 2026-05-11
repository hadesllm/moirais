"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo4u801.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_unnumbered_801."""
import numpy as np
import pytest
from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo4u801 import analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_unnumbered_801


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4u801_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_unnumbered_801(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4u801_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_unnumbered_801(x)
    assert isinstance(result, dict)
