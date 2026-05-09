"""Tests for moirais.fn.nebzr -- RRT planner."""

from moirais.fn.nebzr import rrt_plan, nebzr
from moirais.fn._containers import DescriptiveResult


class TestNebzr:
    def test_alias(self):
        assert nebzr is rrt_plan

    def test_finds_path(self):
        result = rrt_plan(start=(0, 0), goal=(5, 5), bounds=(0, 0, 6, 6), seed=42)
        assert isinstance(result, DescriptiveResult)
        assert result.value < float("inf")
        assert result.extra["found"]

    def test_with_obstacles(self):
        result = rrt_plan(
            start=(0, 0), goal=(10, 10),
            bounds=(0, 0, 12, 12),
            obstacles=[(5, 5, 1.0)],
            max_iter=5000, seed=42,
        )
        assert result.extra["tree_size"] > 1
