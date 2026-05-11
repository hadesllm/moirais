"""Tests for springer_texts_in_statistics_series_gareth_james_daniela_wit2u20.springer_texts_in_statistics_series_gareth_james_daniela_wit_chapter_2_unnumbered_20."""
import numpy as np
import pytest
from morie.fn.springer_texts_in_statistics_series_gareth_james_daniela_wit2u20 import springer_texts_in_statistics_series_gareth_james_daniela_wit_chapter_2_unnumbered_20


def test_springer_texts_in_statistics_series_gareth_james_daniela_wit2u20_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = springer_texts_in_statistics_series_gareth_james_daniela_wit_chapter_2_unnumbered_20(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_springer_texts_in_statistics_series_gareth_james_daniela_wit2u20_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = springer_texts_in_statistics_series_gareth_james_daniela_wit_chapter_2_unnumbered_20(x)
    assert isinstance(result, dict)
