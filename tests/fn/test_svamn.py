"""Tests for moirais.fn.svamn -- Sequential amendment procedure"""

import numpy as np
import pytest

from moirais.fn.svamn import amendment_seq


class TestAmendmentSeq:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = amendment_seq(data)
        assert result.value is not None

    def test_output_type(self):
        result = amendment_seq(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
