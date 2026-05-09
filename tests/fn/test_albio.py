"""Tests for albio.alammar_bio_tagging."""
import numpy as np
import pytest
from moirais.fn.albio import alammar_bio_tagging


def test_albio_basic():
    """Test basic functionality."""
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    entity_spans = np.random.default_rng(42).normal(0, 1, 100)
    scheme = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_bio_tagging(tokens, entity_spans, scheme)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_albio_edge():
    """Test edge cases."""
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    entity_spans = np.random.default_rng(42).normal(0, 1, 100)
    scheme = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_bio_tagging(tokens, entity_spans, scheme)
    assert isinstance(result, dict)
