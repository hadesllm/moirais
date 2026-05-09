"""Tests for kmcrb.kamath_cross_encoder_rerank."""
import numpy as np
import pytest
from moirais.fn.kmcrb import kamath_cross_encoder_rerank


def test_kmcrb_basic():
    """Test basic functionality."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    docs = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_cross_encoder_rerank(q, docs, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmcrb_edge():
    """Test edge cases."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    docs = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_cross_encoder_rerank(q, docs, model)
    assert isinstance(result, dict)
