"""Tests for Itakura-Saito distance."""
import numpy as np
import pytest
from moirais.fn.itakr import itakura_saito, itakr


def test_same():
    p = np.array([1.0, 2.0, 3.0])
    r = itakura_saito(p, p)
    assert r.estimate == pytest.approx(0.0, abs=1e-10)


def test_positive():
    p = np.array([1.0, 2.0, 3.0])
    q = np.array([1.5, 1.5, 1.5])
    r = itakura_saito(p, q)
    assert r.estimate > 0


def test_alias():
    assert itakr is itakura_saito


def test_zero_raises():
    with pytest.raises(ValueError):
        itakura_saito(np.array([1.0, 0.0]), np.array([1.0, 1.0]))
