"""Tests for moirais.fn.cmptm — compliance timeline."""

import pytest
import pandas as pd
from moirais.fn.cmptm import compliance_timeline
from moirais.fn._containers import DescriptiveResult


class TestComplianceTimeline:

    def test_returns_descriptive(self):
        df = pd.DataFrame({"compliant": [1, 0, 1, 1, 0, 1], "period": [1, 1, 1, 2, 2, 2]})
        result = compliance_timeline(df)
        assert isinstance(result, DescriptiveResult)
        assert len(result.value) == 2

    def test_rates_bounded(self):
        df = pd.DataFrame({"compliant": [1, 0, 1, 0], "period": [1, 1, 2, 2]})
        result = compliance_timeline(df)
        assert (result.value["rate"] >= 0).all() and (result.value["rate"] <= 1).all()
