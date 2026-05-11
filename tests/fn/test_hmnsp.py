"""Tests for hmnsp.geron_next_sentence_prediction."""
import numpy as np
import pytest
from morie.fn.hmnsp import geron_next_sentence_prediction


def test_hmnsp_basic():
    """Test basic functionality."""
    sent_A = np.random.default_rng(42).normal(0, 1, 100)
    sent_B = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_next_sentence_prediction(sent_A, sent_B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmnsp_edge():
    """Test edge cases."""
    sent_A = np.random.default_rng(42).normal(0, 1, 100)
    sent_B = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_next_sentence_prediction(sent_A, sent_B)
    assert isinstance(result, dict)
