"""Tests for lpcep.py - LPC to cepstral coefficients."""
import numpy as np
from moirais.fn.lpcep import lpc_to_cepstral_fn, lpcep


def test_lpcep_returns_result():
    coeffs = np.array([0.5, -0.3, 0.1])
    result = lpc_to_cepstral_fn(coeffs, n_ceps=13)
    assert result.name == "lpc_to_cepstral"
    assert len(result.extra["cepstral_coeffs"]) == 13


def test_lpcep_default_nceps():
    coeffs = np.array([0.5, -0.3])
    result = lpc_to_cepstral_fn(coeffs)
    assert result.extra["n_ceps"] == 13


def test_lpcep_alias():
    coeffs = np.array([0.3])
    result = lpcep(coeffs, n_ceps=5)
    assert result.name == "lpc_to_cepstral"
