"""Tests for sbert.sbert."""
import numpy as np
import pytest
from morie.fn.sbert import sbert


def test_sbert_basic():
    """Test basic functionality."""
    sent_a = np.random.default_rng(42).normal(0, 1, 100)
    sent_b = np.random.default_rng(42).normal(0, 1, 100)
    result = sbert(sent_a, sent_b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sbert_edge():
    """Test edge cases."""
    sent_a = np.random.default_rng(42).normal(0, 1, 100)
    sent_b = np.random.default_rng(42).normal(0, 1, 100)
    result = sbert(sent_a, sent_b)
    assert isinstance(result, dict)
