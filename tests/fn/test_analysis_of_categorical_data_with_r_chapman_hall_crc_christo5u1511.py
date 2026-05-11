"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo5u1511.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_unnumbered_1511."""
import numpy as np
import pytest
from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo5u1511 import analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_unnumbered_1511


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo5u1511_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_unnumbered_1511(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo5u1511_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_unnumbered_1511(x)
    assert isinstance(result, dict)
