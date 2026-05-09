"""Test alnmt."""
import numpy as np
from moirais.fn.alnmt import alpha_nominate_score


def test_alnmt_basic():
    r = alpha_nominate_score()
    assert r.value is not None


def test_alnmt_name():
    r = alpha_nominate_score()
    assert r.name
