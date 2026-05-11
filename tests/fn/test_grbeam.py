"""Tests for grbeam.geron_beam_search_decoder."""
import numpy as np
import pytest
from morie.fn.grbeam import geron_beam_search_decoder


def test_grbeam_basic():
    """Test basic functionality."""
    scores = np.random.default_rng(42).uniform(0, 1, 100)
    beam_width = np.random.default_rng(42).normal(0, 1, 100)
    max_len = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_beam_search_decoder(scores, beam_width, max_len)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grbeam_edge():
    """Test edge cases."""
    scores = np.random.default_rng(42).uniform(0, 1, 100)
    beam_width = np.random.default_rng(42).normal(0, 1, 100)
    max_len = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_beam_search_decoder(scores, beam_width, max_len)
    assert isinstance(result, dict)
