"""Tests for moirais.fn.svppm -- Party manifesto scaling (Wordscores)"""

import numpy as np
import pytest

from moirais.fn.svppm import party_manifesto


class TestPartyManifesto:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = party_manifesto(data)
        assert result.value is not None

    def test_output_type(self):
        result = party_manifesto(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
