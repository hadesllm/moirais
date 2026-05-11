"""Tests for morie.fn.strta -- Stratified 2x2xK analysis."""

import pytest
from morie.fn.strta import stratified_analysis


class TestStratifiedAnalysis:
    def test_homogeneous(self):
        tables = [(20, 80, 10, 90), (30, 70, 15, 85)]
        res = stratified_analysis(tables)
        assert res.measure == "stratified_2x2xK"
        assert res.estimate > 0

    def test_breslow_day(self):
        tables = [(20, 80, 10, 90), (30, 70, 15, 85)]
        res = stratified_analysis(tables)
        assert "breslow_day_p" in res.extra

    def test_too_few(self):
        with pytest.raises(ValueError):
            stratified_analysis([(10, 10, 10, 10)])
