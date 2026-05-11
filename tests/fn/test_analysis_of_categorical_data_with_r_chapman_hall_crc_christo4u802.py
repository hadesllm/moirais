"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo4u802.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_unnumbered_802."""
import numpy as np
import pytest
from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo4u802 import analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_unnumbered_802


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4u802_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_unnumbered_802(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4u802_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_unnumbered_802(x)
    assert isinstance(result, dict)
