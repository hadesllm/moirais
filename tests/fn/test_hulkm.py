"""You have power over your mind — not outside events. — Marcus Aurelius"""

import numpy as np
import pytest
from moirais.fn.hulkm import lot_acceptance, hulkm
from moirais.fn._containers import DescriptiveResult


class TestHulkm:
    def test_alias(self):
        assert hulkm is lot_acceptance

    def test_accept_lot(self):
        rng = np.random.default_rng(42)
        sample = rng.normal(50, 1, 30)
        r = lot_acceptance(sample, spec_lower=45, spec_upper=55)
        assert isinstance(r, DescriptiveResult)
        assert r.value["accept"] is True
        assert r.value["cpk"] > 1.0

    def test_reject_lot(self):
        sample = np.array([100, 101, 102, 99, 98])
        r = lot_acceptance(sample, spec_upper=50)
        assert r.value["accept"] is False
