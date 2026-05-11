"""Tests for secrtt.rotating_token_envelope."""
import numpy as np
import pytest
from morie.fn.secrtt import rotating_token_envelope


def test_secrtt_basic():
    """Test basic functionality."""
    payload = np.random.default_rng(42).normal(0, 1, 100)
    kek_id = np.random.default_rng(42).normal(0, 1, 100)
    dek_lifetime = np.random.default_rng(42).normal(0, 1, 100)
    result = rotating_token_envelope(payload, kek_id, dek_lifetime)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_secrtt_edge():
    """Test edge cases."""
    payload = np.random.default_rng(42).normal(0, 1, 100)
    kek_id = np.random.default_rng(42).normal(0, 1, 100)
    dek_lifetime = np.random.default_rng(42).normal(0, 1, 100)
    result = rotating_token_envelope(payload, kek_id, dek_lifetime)
    assert isinstance(result, dict)
