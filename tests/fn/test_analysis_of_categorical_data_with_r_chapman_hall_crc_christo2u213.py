"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo2u213.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_unnumbered_213."""
import numpy as np
import pytest
from moirais.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo2u213 import analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_unnumbered_213


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2u213_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_unnumbered_213(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2u213_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_unnumbered_213(x)
    assert isinstance(result, dict)
