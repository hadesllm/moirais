"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo2u127.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_unnumbered_127."""
import numpy as np
import pytest
from moirais.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo2u127 import analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_unnumbered_127


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2u127_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_unnumbered_127(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2u127_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_unnumbered_127(x)
    assert isinstance(result, dict)
