"""Tests for secarg.argon2id_kdf."""
import numpy as np
import pytest
from moirais.fn.secarg import argon2id_kdf


def test_secarg_basic():
    """Test basic functionality."""
    password = np.random.default_rng(42).normal(0, 1, 100)
    salt = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    t = np.linspace(0, 10, 100)
    p = 5
    result = argon2id_kdf(password, salt, m, t, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_secarg_edge():
    """Test edge cases."""
    password = np.random.default_rng(42).normal(0, 1, 100)
    salt = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    t = np.linspace(0, 10, 100)
    p = 5
    result = argon2id_kdf(password, salt, m, t, p)
    assert isinstance(result, dict)
