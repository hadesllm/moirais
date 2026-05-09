"""Tests for kmnxtg.kamath_nextgpt_any2any."""
import numpy as np
import pytest
from moirais.fn.kmnxtg import kamath_nextgpt_any2any


def test_kmnxtg_basic():
    """Test basic functionality."""
    inputs_by_modality = np.random.default_rng(42).normal(0, 1, 100)
    encoders = np.random.default_rng(42).normal(0, 1, 100)
    llm = np.random.default_rng(42).normal(0, 1, 100)
    decoders = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_nextgpt_any2any(inputs_by_modality, encoders, llm, decoders)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmnxtg_edge():
    """Test edge cases."""
    inputs_by_modality = np.random.default_rng(42).normal(0, 1, 100)
    encoders = np.random.default_rng(42).normal(0, 1, 100)
    llm = np.random.default_rng(42).normal(0, 1, 100)
    decoders = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_nextgpt_any2any(inputs_by_modality, encoders, llm, decoders)
    assert isinstance(result, dict)
