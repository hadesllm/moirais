"""Tests for morie.fn.plidl -- plot ideal and stimuli."""

import numpy as np
from morie.fn.plidl import plot_ideal_and_stimuli, plidl


def test_plidl_smoke():
    Xr = np.array([[0, 0], [1, 1]], dtype=float)
    Xs = np.array([[0.5, 0.5]], dtype=float)
    r = plidl(Xr, Xs)
    assert r.name == "plot_ideal_and_stimuli"
    assert r.extra["n_resp"] == 2
    assert r.extra["n_stim"] == 1


def test_plidl_alias():
    assert plidl is plot_ideal_and_stimuli
