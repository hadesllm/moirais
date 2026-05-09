"""Tests for moirais.fn.strfr -- Luminosity function."""

import numpy as np
from moirais.fn.strfr import luminosity_function, strfr
from moirais.fn._containers import DescriptiveResult


class TestStrfr:
    def test_alias(self):
        assert strfr is luminosity_function

    def test_basic(self):
        rng = np.random.default_rng(42)
        mags = rng.normal(-20, 2, 100)
        result = luminosity_function(mags, bin_width=1.0, volume=100.0)
        assert isinstance(result, DescriptiveResult)
        assert len(result.value["bin_centers"]) > 0

    def test_single_bin(self):
        mags = [5.0, 5.1, 5.2, 4.9, 5.0]
        result = luminosity_function(mags, bin_width=10.0)
        assert len(result.value["phi"]) >= 1
