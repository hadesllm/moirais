"""Tests for moirais.fn.amscl — Aldrich-McKelvey scaling."""
import numpy as np
import pytest

from moirais.fn.amscl import amscl


def test_amscl_smoke():
    Z = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 5.0], [1.5, 3.5, 4.5], [0.5, 2.5, 4.0]])
    r = amscl(Z)
    assert r.name == "aldrich_mckelvey_scaling"
    assert "zhat" in r.extra


def test_cheatsheet():
    from moirais.fn.amscl import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
