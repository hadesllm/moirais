"""Test cllbl."""
import numpy as np
import pytest
from morie.fn.cllbl import cllbl


def test_cllbl_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = cllbl(data=data, n=30, k=3)
    assert r.value is not None


def test_cllbl_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = cllbl(data=data, n=30, k=3)
    assert r.name
