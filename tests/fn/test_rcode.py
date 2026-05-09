"""Tests for moirais.fn.rcode — recode responses."""
import numpy as np
from moirais.fn.rcode import rcode


def test_rcode_smoke():
    data = [[1, 2, 99], [77, 3, 4]]
    r = rcode(data)
    assert r.name == "recode_responses"
    assert r.value == 2
    assert np.isnan(r.extra["cleaned"][0][2])
    assert np.isnan(r.extra["cleaned"][1][0])


def test_cheatsheet():
    from moirais.fn.rcode import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
