"""Tests for morie.fn.lrfor -- Forensic likelihood ratio."""

import numpy as np
import pytest
from morie.fn.lrfor import lrfor


class TestLrfor:
    def test_lr_is_inverse_rmp(self):
        res = lrfor([(0.1, 0.1)])
        lr = res.extra["LR"]
        assert lr == pytest.approx(1.0 / 0.01, rel=1e-6)

    def test_defense_hypothesis(self):
        res_p = lrfor([(0.1, 0.1)])
        res_d = lrfor([(0.1, 0.1)], hypothesis="defense")
        assert res_d.extra["LR"] == pytest.approx(1.0 / res_p.extra["LR"], rel=1e-6)

    def test_multi_locus(self):
        res = lrfor([(0.1, 0.1), (0.2, 0.2)])
        assert res.extra["LR"] > 1.0

    def test_theta_correction(self):
        res_no = lrfor([(0.1, 0.2)])
        res_th = lrfor([(0.1, 0.2)], theta=0.03)
        assert res_th.extra["LR"] < res_no.extra["LR"]
