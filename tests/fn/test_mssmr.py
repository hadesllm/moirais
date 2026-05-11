"""Tests for morie.fn.mssmr -- Replicated SMACOF"""

import numpy as np
import pytest

from morie.fn.mssmr import smacof_replicate


class TestSmacofReplicate:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = smacof_replicate(data)
        assert result.value is not None

    def test_output_type(self):
        result = smacof_replicate(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
