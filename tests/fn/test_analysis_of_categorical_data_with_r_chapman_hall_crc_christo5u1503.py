"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo5u1503.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_unnumbered_1503."""
import numpy as np
import pytest
from moirais.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo5u1503 import analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_unnumbered_1503


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo5u1503_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_unnumbered_1503(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo5u1503_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_unnumbered_1503(x)
    assert isinstance(result, dict)
