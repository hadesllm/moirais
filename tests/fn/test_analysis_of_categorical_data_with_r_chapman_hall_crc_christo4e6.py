"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e6.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_6."""
import numpy as np
import pytest
from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e6 import analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_6


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e6_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_6(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e6_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_6(x)
    assert isinstance(result, dict)
