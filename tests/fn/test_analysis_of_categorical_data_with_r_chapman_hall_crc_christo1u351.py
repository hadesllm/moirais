"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo1u351.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_unnumbered_351."""
import numpy as np
import pytest
from moirais.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo1u351 import analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_unnumbered_351


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1u351_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_unnumbered_351(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1u351_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_unnumbered_351(x)
    assert isinstance(result, dict)
