"""Tests for moirais.fn.amci — A-M confidence intervals."""
import numpy as np
from moirais.fn.amci import amci


def test_amci_smoke():
    rng = np.random.default_rng(42)
    boot = rng.standard_normal((200, 3))
    r = amci(boot, alpha=0.05)
    assert r.name == "am_confidence_intervals"
    assert r.value == 0.95
    assert len(r.extra["lower"]) == 3
    assert len(r.extra["upper"]) == 3


def test_cheatsheet():
    from moirais.fn.amci import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
