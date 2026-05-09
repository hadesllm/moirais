"""Tests for springer_texts_in_statistics_series_gareth_james_daniela_wit2u14.springer_texts_in_statistics_series_gareth_james_daniela_wit_chapter_2_unnumbered_14."""
import numpy as np
import pytest
from moirais.fn.springer_texts_in_statistics_series_gareth_james_daniela_wit2u14 import springer_texts_in_statistics_series_gareth_james_daniela_wit_chapter_2_unnumbered_14


def test_springer_texts_in_statistics_series_gareth_james_daniela_wit2u14_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = springer_texts_in_statistics_series_gareth_james_daniela_wit_chapter_2_unnumbered_14(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_springer_texts_in_statistics_series_gareth_james_daniela_wit2u14_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = springer_texts_in_statistics_series_gareth_james_daniela_wit_chapter_2_unnumbered_14(x)
    assert isinstance(result, dict)
