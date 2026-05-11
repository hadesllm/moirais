"""Test fftsi."""
import numpy as np
import pytest
from morie.fn.fftsi import fftsi


def test_fftsi_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = fftsi(points=pts, n=40)
    assert r.value is not None


def test_fftsi_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = fftsi(points=pts, n=40)
    assert r.name
