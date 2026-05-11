"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo1u360.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_unnumbered_360."""
import numpy as np
import pytest
from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo1u360 import analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_unnumbered_360


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1u360_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_unnumbered_360(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1u360_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_unnumbered_360(x)
    assert isinstance(result, dict)
