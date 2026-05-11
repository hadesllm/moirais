"""Tests for bidmn.py - Bidomain cardiac model."""
import numpy as np
from morie.fn.bidmn import bidomain_model_fn, bidmn


def test_bidmn_returns_descriptive_result():
    result = bidomain_model_fn(n_cells=20, duration=5.0, dt=0.1)
    assert result.name == "bidomain"
    assert result.value == 20


def test_bidmn_voltage_matrix_shape():
    result = bidomain_model_fn(n_cells=10, duration=5.0, dt=0.1)
    V = result.extra["voltage_matrix"]
    assert V.shape == (50, 10)


def test_bidmn_initial_stimulus():
    result = bidomain_model_fn(n_cells=10, duration=2.0, dt=0.1)
    V = result.extra["voltage_matrix"]
    assert V[0, 0] == 20.0
    assert V[0, 5] == -85.0


def test_bidmn_alias():
    result = bidmn(n_cells=5, duration=2.0, dt=0.5)
    assert result.name == "bidomain"
