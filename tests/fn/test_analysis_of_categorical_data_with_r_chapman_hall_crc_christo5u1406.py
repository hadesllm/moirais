"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo5u1406.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_unnumbered_1406."""
import numpy as np
import pytest
from moirais.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo5u1406 import analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_unnumbered_1406


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo5u1406_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_unnumbered_1406(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo5u1406_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_unnumbered_1406(x)
    assert isinstance(result, dict)
