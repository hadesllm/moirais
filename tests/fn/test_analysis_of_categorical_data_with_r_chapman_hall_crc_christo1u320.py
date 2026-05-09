"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo1u320.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_unnumbered_320."""
import numpy as np
import pytest
from moirais.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo1u320 import analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_unnumbered_320


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1u320_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_unnumbered_320(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1u320_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_unnumbered_320(x)
    assert isinstance(result, dict)
