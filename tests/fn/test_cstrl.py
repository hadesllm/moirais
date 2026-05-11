"""Tests for morie.fn.cstrl — custody release type."""

import pytest
import numpy as np
from morie.fn.cstrl import custody_release_type
from morie.fn._containers import DescriptiveResult


class TestCustodyReleaseType:

    def test_returns_descriptive(self):
        rt = np.array(["parole", "statutory", "warrant", "parole"])
        result = custody_release_type(rt)
        assert isinstance(result, DescriptiveResult)
        assert sum(result.extra["proportions"].values()) == pytest.approx(1.0)

    def test_single_type(self):
        result = custody_release_type(np.array(["parole", "parole"]))
        assert len(result.extra["distribution"]) == 1
