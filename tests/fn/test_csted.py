"""Tests for morie.fn.csted — custody education."""

import pytest
import pandas as pd
from morie.fn.csted import custody_education
from morie.fn._containers import DescriptiveResult


class TestCustodyEducation:

    def test_returns_descriptive(self):
        df = pd.DataFrame({"edu_enrolled": [1, 1, 0, 1], "edu_completed": [1, 0, 0, 1]})
        result = custody_education(df)
        assert isinstance(result, DescriptiveResult)

    def test_completion_rate(self):
        df = pd.DataFrame({"edu_enrolled": [1, 1, 1, 0], "edu_completed": [1, 1, 0, 0]})
        result = custody_education(df)
        assert result.extra["completion_rate"] == pytest.approx(2 / 3)
