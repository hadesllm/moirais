"""Tests for morie.fn.simpe — simulate perceptions."""
import numpy as np
from morie.fn.simpe import simpe


def test_simpe_smoke():
    r = simpe([1.0, 3.0, 5.0], n_resp=10, sigma=0.3)
    assert r.name == "simulate_perceptions"
    assert r.value == 10
    assert len(r.extra["perceptions"]) == 10
    assert r.extra["n_stimuli"] == 3


def test_cheatsheet():
    from morie.fn.simpe import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
