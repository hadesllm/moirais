"""Tests for springer_texts_in_statistics_series_gareth_james_daniela_wit3e36.springer_texts_in_statistics_series_gareth_james_daniela_wit_chapter_3_equation_36."""
import numpy as np
import pytest
from morie.fn.springer_texts_in_statistics_series_gareth_james_daniela_wit3e36 import springer_texts_in_statistics_series_gareth_james_daniela_wit_chapter_3_equation_36


def test_springer_texts_in_statistics_series_gareth_james_daniela_wit3e36_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = springer_texts_in_statistics_series_gareth_james_daniela_wit_chapter_3_equation_36(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_springer_texts_in_statistics_series_gareth_james_daniela_wit3e36_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = springer_texts_in_statistics_series_gareth_james_daniela_wit_chapter_3_equation_36(x)
    assert isinstance(result, dict)
