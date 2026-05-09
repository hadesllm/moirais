"""Tests for sacrb.sacrebleu."""
import numpy as np
import pytest
from moirais.fn.sacrb import sacrebleu


def test_sacrb_basic():
    """Test basic functionality."""
    candidate = np.random.default_rng(42).normal(0, 1, 100)
    references = np.random.default_rng(42).normal(0, 1, 100)
    result = sacrebleu(candidate, references)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sacrb_edge():
    """Test edge cases."""
    candidate = np.random.default_rng(42).normal(0, 1, 100)
    references = np.random.default_rng(42).normal(0, 1, 100)
    result = sacrebleu(candidate, references)
    assert isinstance(result, dict)
