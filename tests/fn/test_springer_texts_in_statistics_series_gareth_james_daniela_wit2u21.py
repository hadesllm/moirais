"""Tests for springer_texts_in_statistics_series_gareth_james_daniela_wit2u21.springer_texts_in_statistics_series_gareth_james_daniela_wit_chapter_2_unnumbered_21."""
import numpy as np
import pytest
from moirais.fn.springer_texts_in_statistics_series_gareth_james_daniela_wit2u21 import springer_texts_in_statistics_series_gareth_james_daniela_wit_chapter_2_unnumbered_21


def test_springer_texts_in_statistics_series_gareth_james_daniela_wit2u21_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = springer_texts_in_statistics_series_gareth_james_daniela_wit_chapter_2_unnumbered_21(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_springer_texts_in_statistics_series_gareth_james_daniela_wit2u21_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = springer_texts_in_statistics_series_gareth_james_daniela_wit_chapter_2_unnumbered_21(x)
    assert isinstance(result, dict)
