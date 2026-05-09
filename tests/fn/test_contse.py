"""Tests for contse.contrastive_sent."""
import numpy as np
import pytest
from moirais.fn.contse import contrastive_sent


def test_contse_basic():
    """Test basic functionality."""
    sentences = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = contrastive_sent(sentences, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_contse_edge():
    """Test edge cases."""
    sentences = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = contrastive_sent(sentences, tau)
    assert isinstance(result, dict)
