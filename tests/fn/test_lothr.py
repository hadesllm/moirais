"""Tests for moirais.fn.lothr -- template matching."""

import numpy as np
from moirais.fn.lothr import template_match, lothr
from moirais.fn._containers import DescriptiveResult


class TestLothr:
    def test_alias(self):
        assert lothr is template_match

    def test_1d_exact(self):
        tpl = np.array([0, 1, 2, 3, 2, 1, 0], dtype=float)
        sig = np.zeros(50)
        sig[20:27] = tpl
        result = template_match(sig, tpl, method="ncc")
        assert isinstance(result, DescriptiveResult)
        assert result.value > 0.9
        assert result.extra["position"] == 20

    def test_2d_ssd(self):
        rng = np.random.default_rng(42)
        img = rng.normal(0, 1, (20, 20))
        tpl = img[5:10, 5:10].copy()
        result = template_match(img, tpl, method="ssd")
        pos = result.extra["position"]
        assert pos == [5, 5]
