"""Tests for moirais.fn.spkpf -- single-peaked check."""

import numpy as np
from moirais.fn.spkpf import single_peaked_check, spkpf


def test_spkpf_true():
    prefs = np.array([[1, 3, 5, 3, 1]], dtype=float)
    r = spkpf(prefs)
    assert r.name == "single_peaked_check"
    assert r.value is True


def test_spkpf_false():
    prefs = np.array([[1, 5, 2, 5, 1]], dtype=float)
    r = spkpf(prefs)
    assert r.value is False


def test_spkpf_alias():
    assert spkpf is single_peaked_check
