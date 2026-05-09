"""Test emirt."""
import numpy as np
from moirais.fn.emirt import em_irt_estimate


def test_emirt_basic():
    r = em_irt_estimate()
    assert r.value is not None


def test_emirt_name():
    r = em_irt_estimate()
    assert r.name
