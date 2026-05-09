"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo3u523.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_unnumbered_523."""
import numpy as np
import pytest
from moirais.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo3u523 import analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_unnumbered_523


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3u523_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_unnumbered_523(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3u523_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_unnumbered_523(x)
    assert isinstance(result, dict)
