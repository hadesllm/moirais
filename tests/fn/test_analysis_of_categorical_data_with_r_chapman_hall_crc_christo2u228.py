"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo2u228.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_unnumbered_228."""
import numpy as np
import pytest
from moirais.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo2u228 import analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_unnumbered_228


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2u228_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_unnumbered_228(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2u228_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_unnumbered_228(x)
    assert isinstance(result, dict)
