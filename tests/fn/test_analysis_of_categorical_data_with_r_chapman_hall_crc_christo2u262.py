"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo2u262.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_unnumbered_262."""
import numpy as np
import pytest
from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo2u262 import analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_unnumbered_262


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2u262_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_unnumbered_262(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2u262_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_unnumbered_262(x)
    assert isinstance(result, dict)
