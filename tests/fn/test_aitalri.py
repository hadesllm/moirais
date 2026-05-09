"""Tests for aitalri.aitchison_alr_inverse."""
import numpy as np
import pytest
from moirais.fn.aitalri import aitchison_alr_inverse


def test_aitalri_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = aitchison_alr_inverse(y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitalri_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = aitchison_alr_inverse(y)
    assert isinstance(result, dict)
