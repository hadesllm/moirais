"""Benchmarking and performance analysis for MOIRAIS.

Provides comprehensive tools for profiling module execution, tracking memory
usage, detecting performance regressions, and generating benchmark reports.
All benchmarks capture system metadata for reproducibility.

The benchmarking framework is designed for scientific computing workloads
where both computational cost and statistical output quality matter.
Integration with CodeCarbon emissions tracking is supported through the
``moirais[carbon]`` optional dependency.

References
----------
Boisvert, R. F. et al. (2001). *The Architecture of Scientific Software*
(Vol. 60). Springer. https://doi.org/10.1007/978-1-4615-1339-4

Kalibera, T. & Jones, R. (2013). Rigorous benchmarking in reasonable time.
*Proceedings of the 2013 International Symposium on Memory Management*,
63--74. https://doi.org/10.1145/2464157.2464160
"""

from __future__ import annotations

import gc
import json
import logging
import os
import platform
import sys
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SystemInfo:
    """Host system information captured at benchmark time."""

    python_version: str
    platform: str
    machine: str
    cpu_count: int
    total_memory_gb: float
    os_version: str
    timestamp: str
    package_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "machine": self.machine,
            "cpu_count": self.cpu_count,
            "total_memory_gb": self.total_memory_gb,
            "os_version": self.os_version,
            "timestamp": self.timestamp,
            "package_versions": self.package_versions,
        }


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    iterations: int
    times: list[float]  # seconds per iteration
    peak_memory_mb: float
    output_size_bytes: int = 0
    system_info: SystemInfo | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_time(self) -> float:
        """Mean execution time in seconds."""
        return float(np.mean(self.times)) if self.times else 0.0

    @property
    def median_time(self) -> float:
        """Median execution time."""
        return float(np.median(self.times)) if self.times else 0.0

    @property
    def std_time(self) -> float:
        """Standard deviation of execution time."""
        return float(np.std(self.times, ddof=1)) if len(self.times) > 1 else 0.0

    @property
    def min_time(self) -> float:
        """Minimum execution time."""
        return float(np.min(self.times)) if self.times else 0.0

    @property
    def max_time(self) -> float:
        """Maximum execution time."""
        return float(np.max(self.times)) if self.times else 0.0

    @property
    def ci_95(self) -> tuple[float, float]:
        """95% confidence interval for mean time (normal approximation)."""
        if len(self.times) < 2:
            return (self.mean_time, self.mean_time)
        se = self.std_time / np.sqrt(len(self.times))
        return (self.mean_time - 1.96 * se, self.mean_time + 1.96 * se)

    @property
    def cv(self) -> float:
        """Coefficient of variation."""
        if self.mean_time == 0:
            return 0.0
        return self.std_time / self.mean_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        ci = self.ci_95
        return {
            "name": self.name,
            "iterations": self.iterations,
            "mean_time_s": self.mean_time,
            "median_time_s": self.median_time,
            "std_time_s": self.std_time,
            "min_time_s": self.min_time,
            "max_time_s": self.max_time,
            "ci_95_lower": ci[0],
            "ci_95_upper": ci[1],
            "cv": self.cv,
            "peak_memory_mb": self.peak_memory_mb,
            "output_size_bytes": self.output_size_bytes,
            "metadata": self.metadata,
            "system_info": self.system_info.to_dict() if self.system_info else None,
        }


@dataclass
class MemoryProfile:
    """Memory usage profile over time."""

    name: str
    peak_mb: float
    snapshots: list[tuple[float, float]]  # (time_offset_s, memory_mb)
    traceback_peak: str = ""

    def to_dataframe(self) -> pd.DataFrame:
        """Convert snapshots to a DataFrame."""
        if not self.snapshots:
            return pd.DataFrame(columns=["time_s", "memory_mb"])
        return pd.DataFrame(self.snapshots, columns=["time_s", "memory_mb"])


@dataclass
class BenchmarkComparison:
    """Comparison of two benchmark results."""

    name_a: str
    name_b: str
    result_a: BenchmarkResult
    result_b: BenchmarkResult
    speedup: float  # >1 means B is faster
    memory_ratio: float  # >1 means B uses more memory
    statistically_significant: bool
    p_value: float

    def summary(self) -> str:
        """Human-readable summary."""
        direction = "faster" if self.speedup > 1 else "slower"
        sig = " (significant)" if self.statistically_significant else " (not significant)"
        return f"{self.name_b} is {abs(self.speedup):.2f}x {direction} than {self.name_a}{sig}, p = {self.p_value:.4f}"


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    name: str
    results: list[BenchmarkResult] = field(default_factory=list)
    timestamp: str = ""
    system_info: SystemInfo | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all results to a summary DataFrame."""
        rows = []
        for r in self.results:
            rows.append(
                {
                    "name": r.name,
                    "mean_time_s": r.mean_time,
                    "median_time_s": r.median_time,
                    "std_time_s": r.std_time,
                    "min_time_s": r.min_time,
                    "max_time_s": r.max_time,
                    "peak_memory_mb": r.peak_memory_mb,
                    "iterations": r.iterations,
                    "cv": r.cv,
                }
            )
        return pd.DataFrame(rows)


@dataclass
class RegressionReport:
    """Performance regression detection report."""

    comparisons: list[BenchmarkComparison]
    regressions: list[BenchmarkComparison]
    improvements: list[BenchmarkComparison]
    threshold_pct: float

    @property
    def has_regressions(self) -> bool:
        """True if any significant regressions detected."""
        return len(self.regressions) > 0


# ---------------------------------------------------------------------------
# System information
# ---------------------------------------------------------------------------


def capture_system_info() -> SystemInfo:
    """Capture current system information for benchmark reproducibility.

    Returns
    -------
    SystemInfo
        System metadata including Python version, OS, CPU, RAM, and
        key package versions.
    """
    import importlib.metadata

    total_mem_gb = 0.0
    try:
        if sys.platform == "darwin":
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                total_mem_gb = int(result.stdout.strip()) / (1024**3)
        elif sys.platform == "linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        total_mem_gb = kb / (1024**2)
                        break
    except Exception:
        total_mem_gb = 0.0

    # Key package versions
    packages = ["pandas", "numpy", "scipy", "scikit-learn", "statsmodels", "DoubleML", "rich"]
    pkg_versions: dict[str, str] = {}
    for pkg in packages:
        try:
            pkg_versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass

    return SystemInfo(
        python_version=sys.version,
        platform=sys.platform,
        machine=platform.machine(),
        cpu_count=os.cpu_count() or 1,
        total_memory_gb=round(total_mem_gb, 2),
        os_version=platform.platform(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        package_versions=pkg_versions,
    )


# ---------------------------------------------------------------------------
# Core benchmarking
# ---------------------------------------------------------------------------


def benchmark(
    func: Callable[..., Any],
    *args: Any,
    name: str | None = None,
    iterations: int = 5,
    warmup: int = 1,
    track_memory: bool = True,
    gc_collect: bool = True,
    **kwargs: Any,
) -> BenchmarkResult:
    """Benchmark a function's execution time and memory usage.

    Parameters
    ----------
    func : callable
        Function to benchmark.
    *args
        Positional arguments passed to *func*.
    name : str | None
        Benchmark name (default: function name).
    iterations : int
        Number of timed iterations (default 5).
    warmup : int
        Number of warmup iterations to discard (default 1).
    track_memory : bool
        If True, track peak memory allocation.
    gc_collect : bool
        If True, run garbage collection before each iteration.
    **kwargs
        Keyword arguments passed to *func*.

    Returns
    -------
    BenchmarkResult
    """
    if name is None:
        name = getattr(func, "__name__", str(func))

    logger.info("Benchmarking '%s': %d iterations + %d warmup", name, iterations, warmup)

    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Timed iterations
    times: list[float] = []
    peak_mem = 0.0

    if track_memory:
        tracemalloc.start()

    for i in range(iterations):
        if gc_collect:
            gc.collect()

        if track_memory:
            tracemalloc.clear_traces()

        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if track_memory:
            current, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / (1024 * 1024)
            peak_mem = max(peak_mem, peak_mb)

        logger.debug("  Iteration %d: %.4fs", i + 1, elapsed)

    if track_memory:
        tracemalloc.stop()

    # Estimate output size
    output_size = 0
    if isinstance(result, pd.DataFrame):
        output_size = result.memory_usage(deep=True).sum()
    elif isinstance(result, dict):
        output_size = len(json.dumps(result, default=str).encode())
    elif isinstance(result, (str, bytes)):
        output_size = len(result) if isinstance(result, bytes) else len(result.encode())

    sys_info = capture_system_info()

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        times=times,
        peak_memory_mb=round(peak_mem, 2),
        output_size_bytes=output_size,
        system_info=sys_info,
    )


def benchmark_module(
    module_name: str,
    *,
    cpads_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    iterations: int = 3,
    warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark a single MOIRAIS module.

    Parameters
    ----------
    module_name : str
        Name of the module to benchmark (from MODULE_SPECS).
    cpads_csv : str | Path | None
        Path to CPADS CSV file.
    output_dir : str | Path | None
        Output directory for module results.
    iterations : int
        Number of iterations.
    warmup : int
        Number of warmup iterations.

    Returns
    -------
    BenchmarkResult
    """
    import tempfile

    from .modules import run_module

    _tmp_ctx = None
    if output_dir is None:
        _tmp_ctx = tempfile.TemporaryDirectory(prefix="moirais-bench-")
        output_dir = _tmp_ctx.name

    try:

        def _run() -> dict[str, Any]:
            return run_module(
                module_name,
                cpads_csv=str(cpads_csv) if cpads_csv else None,
                output_dir=str(output_dir),
            )

        result = benchmark(
            _run,
            name=f"module:{module_name}",
            iterations=iterations,
            warmup=warmup,
        )

        # Measure output size
        out_path = Path(output_dir)
        total_size = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file())
        result.output_size_bytes = total_size
        result.metadata["module_name"] = module_name
        result.metadata["output_dir"] = str(output_dir)

        return result
    finally:
        if _tmp_ctx is not None:
            _tmp_ctx.cleanup()


def benchmark_all_modules(
    *,
    cpads_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    iterations: int = 2,
) -> BenchmarkSuite:
    """Benchmark all MOIRAIS modules.

    Parameters
    ----------
    cpads_csv : str | Path | None
        CPADS CSV path.
    output_dir : str | Path | None
        Base output directory.
    iterations : int
        Iterations per module.

    Returns
    -------
    BenchmarkSuite
    """
    from .modules import MODULE_SPECS

    suite = BenchmarkSuite(
        name="all_modules",
        timestamp=datetime.now(timezone.utc).isoformat(),
        system_info=capture_system_info(),
    )

    for module_name in MODULE_SPECS:
        try:
            r = benchmark_module(
                module_name,
                cpads_csv=cpads_csv,
                output_dir=output_dir,
                iterations=iterations,
                warmup=0,
            )
            suite.results.append(r)
            logger.info("Module %s: mean=%.2fs, peak_mem=%.1fMB", module_name, r.mean_time, r.peak_memory_mb)
        except Exception as exc:
            logger.warning("Module %s failed: %s", module_name, exc)
            suite.results.append(
                BenchmarkResult(
                    name=f"module:{module_name}",
                    iterations=0,
                    times=[],
                    peak_memory_mb=0.0,
                    metadata={"error": str(exc)},
                )
            )

    return suite


# ---------------------------------------------------------------------------
# Memory profiling
# ---------------------------------------------------------------------------


def memory_profile(
    func: Callable[..., Any],
    *args: Any,
    name: str | None = None,
    interval_seconds: float = 0.1,
    **kwargs: Any,
) -> MemoryProfile:
    """Profile memory usage of a function over time.

    Samples memory allocation at regular intervals during execution.

    Parameters
    ----------
    func : callable
        Function to profile.
    *args
        Positional arguments.
    name : str | None
        Profile name.
    interval_seconds : float
        Sampling interval (default 0.1s).
    **kwargs
        Keyword arguments.

    Returns
    -------
    MemoryProfile
    """
    import threading

    if name is None:
        name = getattr(func, "__name__", str(func))

    tracemalloc.start()
    snapshots: list[tuple[float, float]] = []
    done = threading.Event()
    start_time = time.perf_counter()

    def sampler() -> None:
        while not done.is_set():
            current, _peak = tracemalloc.get_traced_memory()
            elapsed = time.perf_counter() - start_time
            snapshots.append((round(elapsed, 4), round(current / (1024 * 1024), 2)))
            done.wait(interval_seconds)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()

    try:
        func(*args, **kwargs)
    finally:
        done.set()
        thread.join(timeout=2.0)

    _, peak = tracemalloc.get_traced_memory()
    peak_mb = peak / (1024 * 1024)

    # Get peak traceback
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")
    traceback_peak = str(top_stats[0]) if top_stats else ""

    tracemalloc.stop()

    return MemoryProfile(
        name=name,
        peak_mb=round(peak_mb, 2),
        snapshots=snapshots,
        traceback_peak=traceback_peak,
    )


# ---------------------------------------------------------------------------
# CPU profiling
# ---------------------------------------------------------------------------


def cpu_profile(
    func: Callable[..., Any],
    *args: Any,
    name: str | None = None,
    sort_by: str = "cumulative",
    top_n: int = 20,
    **kwargs: Any,
) -> dict[str, Any]:
    """Profile CPU usage per function using cProfile.

    Parameters
    ----------
    func : callable
        Function to profile.
    *args
        Positional arguments.
    name : str | None
        Profile name.
    sort_by : str
        Sort key for results (``cumulative``, ``tottime``, ``calls``).
    top_n : int
        Number of top entries to include.
    **kwargs
        Keyword arguments.

    Returns
    -------
    dict[str, Any]
        Profile results with keys ``total_time``, ``top_functions``,
        ``total_calls``.
    """
    import cProfile
    import io
    import pstats

    if name is None:
        name = getattr(func, "__name__", str(func))

    profiler = cProfile.Profile()
    profiler.enable()

    start = time.perf_counter()
    func(*args, **kwargs)
    elapsed = time.perf_counter() - start

    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats(sort_by)

    # Extract top functions
    top_functions: list[dict[str, Any]] = []
    # Access the stats dict directly
    func_stats = stats.stats  # type: ignore[attr-defined]
    sorted_keys = sorted(
        func_stats.keys(),
        key=lambda k: func_stats[k][3] if sort_by == "cumulative" else func_stats[k][2],
        reverse=True,
    )

    for key in sorted_keys[:top_n]:
        filename, lineno, func_name = key
        cc, nc, tt, ct, _callers = func_stats[key]
        top_functions.append(
            {
                "function": func_name,
                "file": f"{filename}:{lineno}",
                "calls": nc,
                "total_time_s": round(tt, 6),
                "cumulative_time_s": round(ct, 6),
                "per_call_s": round(tt / nc, 6) if nc > 0 else 0.0,
            }
        )

    total_calls = sum(v[1] for v in func_stats.values())

    return {
        "name": name,
        "total_time_s": round(elapsed, 4),
        "total_calls": total_calls,
        "top_functions": top_functions,
    }


# ---------------------------------------------------------------------------
# Scalability analysis
# ---------------------------------------------------------------------------


def scalability_analysis(
    func: Callable[[pd.DataFrame], Any],
    df: pd.DataFrame,
    *,
    sizes: list[int] | None = None,
    fractions: list[float] | None = None,
    iterations: int = 3,
    name: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Analyze how execution time scales with data size.

    Runs a function with increasing subsets of the input data and records
    execution time and memory for each size.

    Parameters
    ----------
    func : callable
        Function that takes a DataFrame as input.
    df : pd.DataFrame
        Full dataset.
    sizes : list[int] | None
        Absolute sample sizes to test.
    fractions : list[float] | None
        Fraction of data to test (e.g. ``[0.1, 0.25, 0.5, 1.0]``).
    iterations : int
        Iterations per size.
    name : str | None
        Benchmark name.
    seed : int
        Random seed for sampling.

    Returns
    -------
    pd.DataFrame
        Columns: ``n``, ``mean_time_s``, ``std_time_s``, ``peak_memory_mb``.
    """
    if name is None:
        name = getattr(func, "__name__", "scalability")

    if sizes is None and fractions is None:
        fractions = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

    if sizes is None:
        n_total = len(df)
        sizes = [max(10, int(f * n_total)) for f in (fractions or [])]

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for n in sizes:
        n = min(n, len(df))
        subset = df.iloc[rng.choice(len(df), size=n, replace=n > len(df))]

        r = benchmark(func, subset, name=f"{name}_n{n}", iterations=iterations, warmup=1)
        rows.append(
            {
                "n": n,
                "mean_time_s": r.mean_time,
                "std_time_s": r.std_time,
                "median_time_s": r.median_time,
                "peak_memory_mb": r.peak_memory_mb,
            }
        )
        logger.info("n=%d: mean=%.4fs, peak_mem=%.1fMB", n, r.mean_time, r.peak_memory_mb)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Comparative benchmarks
# ---------------------------------------------------------------------------


def compare_benchmarks(
    result_a: BenchmarkResult,
    result_b: BenchmarkResult,
    *,
    alpha: float = 0.05,
) -> BenchmarkComparison:
    """Compare two benchmark results with statistical testing.

    Uses a two-sample t-test (Welch's) to determine if the performance
    difference is statistically significant.

    Parameters
    ----------
    result_a : BenchmarkResult
        Baseline benchmark.
    result_b : BenchmarkResult
        Comparison benchmark.
    alpha : float
        Significance level.

    Returns
    -------
    BenchmarkComparison
    """
    from scipy import stats as sp_stats

    speedup = result_a.mean_time / result_b.mean_time if result_b.mean_time > 0 else float("inf")
    memory_ratio = result_b.peak_memory_mb / result_a.peak_memory_mb if result_a.peak_memory_mb > 0 else 0.0

    if len(result_a.times) >= 2 and len(result_b.times) >= 2:
        t_stat, p_value = sp_stats.ttest_ind(
            result_a.times,
            result_b.times,
            equal_var=False,
        )
        significant = p_value < alpha
    else:
        p_value = 1.0
        significant = False

    return BenchmarkComparison(
        name_a=result_a.name,
        name_b=result_b.name,
        result_a=result_a,
        result_b=result_b,
        speedup=round(speedup, 4),
        memory_ratio=round(memory_ratio, 4),
        statistically_significant=significant,
        p_value=round(p_value, 6),
    )


def ab_compare(
    func_a: Callable[..., Any],
    func_b: Callable[..., Any],
    *args: Any,
    name_a: str = "A",
    name_b: str = "B",
    iterations: int = 10,
    warmup: int = 2,
    **kwargs: Any,
) -> BenchmarkComparison:
    """Run A/B comparison between two implementations.

    Parameters
    ----------
    func_a : callable
        First implementation.
    func_b : callable
        Second implementation.
    *args
        Arguments passed to both functions.
    name_a : str
        Name for first implementation.
    name_b : str
        Name for second implementation.
    iterations : int
        Number of iterations per implementation.
    warmup : int
        Warmup iterations.
    **kwargs
        Keyword arguments passed to both functions.

    Returns
    -------
    BenchmarkComparison
    """
    result_a = benchmark(func_a, *args, name=name_a, iterations=iterations, warmup=warmup, **kwargs)
    result_b = benchmark(func_b, *args, name=name_b, iterations=iterations, warmup=warmup, **kwargs)
    return compare_benchmarks(result_a, result_b)


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------


def detect_regressions(
    baseline: BenchmarkSuite,
    current: BenchmarkSuite,
    *,
    threshold_pct: float = 10.0,
    alpha: float = 0.05,
) -> RegressionReport:
    """Detect performance regressions between two benchmark suites.

    Parameters
    ----------
    baseline : BenchmarkSuite
        Baseline benchmarks (e.g. from the previous release).
    current : BenchmarkSuite
        Current benchmarks.
    threshold_pct : float
        Percentage slowdown to flag as regression.
    alpha : float
        Significance level for t-test.

    Returns
    -------
    RegressionReport
    """
    baseline_map = {r.name: r for r in baseline.results}
    current_map = {r.name: r for r in current.results}

    comparisons: list[BenchmarkComparison] = []
    regressions: list[BenchmarkComparison] = []
    improvements: list[BenchmarkComparison] = []

    common_names = set(baseline_map.keys()) & set(current_map.keys())

    for name in sorted(common_names):
        comp = compare_benchmarks(baseline_map[name], current_map[name], alpha=alpha)
        comparisons.append(comp)

        pct_change = (
            ((comp.result_b.mean_time - comp.result_a.mean_time) / comp.result_a.mean_time * 100)
            if comp.result_a.mean_time > 0
            else 0.0
        )

        if pct_change > threshold_pct and comp.statistically_significant:
            regressions.append(comp)
        elif pct_change < -threshold_pct and comp.statistically_significant:
            improvements.append(comp)

    return RegressionReport(
        comparisons=comparisons,
        regressions=regressions,
        improvements=improvements,
        threshold_pct=threshold_pct,
    )


# ---------------------------------------------------------------------------
# Statistical summary
# ---------------------------------------------------------------------------


def summarize_benchmark(result: BenchmarkResult) -> dict[str, Any]:
    """Produce a detailed statistical summary of a benchmark.

    Parameters
    ----------
    result : BenchmarkResult
        Benchmark to summarize.

    Returns
    -------
    dict[str, Any]
        Summary statistics including outlier detection.
    """
    times = np.array(result.times)
    if len(times) == 0:
        return {"name": result.name, "error": "no timing data"}

    ci = result.ci_95
    q1 = float(np.percentile(times, 25))
    q3 = float(np.percentile(times, 75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = int(np.sum((times < lower_fence) | (times > upper_fence)))

    return {
        "name": result.name,
        "n": len(times),
        "mean_s": round(result.mean_time, 6),
        "median_s": round(result.median_time, 6),
        "std_s": round(result.std_time, 6),
        "min_s": round(result.min_time, 6),
        "max_s": round(result.max_time, 6),
        "ci_95_lower": round(ci[0], 6),
        "ci_95_upper": round(ci[1], 6),
        "cv": round(result.cv, 4),
        "q1_s": round(q1, 6),
        "q3_s": round(q3, 6),
        "iqr_s": round(iqr, 6),
        "outliers": outliers,
        "peak_memory_mb": result.peak_memory_mb,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_results(
    results: BenchmarkResult | BenchmarkSuite,
    path: str | Path,
    *,
    format: str = "json",
) -> Path:
    """Save benchmark results to disk.

    Parameters
    ----------
    results : BenchmarkResult | BenchmarkSuite
        Results to save.
    path : str | Path
        Output file path.
    format : str
        ``json`` or ``csv``.

    Returns
    -------
    Path
        Path to saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(results, BenchmarkSuite):
        data = {
            "name": results.name,
            "timestamp": results.timestamp,
            "system_info": results.system_info.to_dict() if results.system_info else None,
            "results": [r.to_dict() for r in results.results],
        }
    else:
        data = results.to_dict()

    if format == "json":
        path = path.with_suffix(".json")
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    elif format == "csv":
        path = path.with_suffix(".csv")
        if isinstance(results, BenchmarkSuite):
            df = results.to_dataframe()
        else:
            df = pd.DataFrame([results.to_dict()])
        df.to_csv(path, index=False)

    logger.info("Saved benchmarks to %s", path)
    return path


def load_results(path: str | Path) -> BenchmarkSuite:
    """Load benchmark results from a JSON file.

    Parameters
    ----------
    path : str | Path
        Path to saved results.

    Returns
    -------
    BenchmarkSuite
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    results: list[BenchmarkResult] = []
    for r in data.get("results", []):
        results.append(
            BenchmarkResult(
                name=r["name"],
                iterations=r.get("iterations", 0),
                times=[],  # Raw times not persisted in summary
                peak_memory_mb=r.get("peak_memory_mb", 0.0),
                output_size_bytes=r.get("output_size_bytes", 0),
                metadata=r.get("metadata", {}),
            )
        )
        # Reconstruct times from summary if possible
        mean_t = r.get("mean_time_s", 0.0)
        n = r.get("iterations", 0)
        if n > 0 and mean_t > 0:
            results[-1].times = [mean_t] * n
            results[-1].iterations = n

    sys_data = data.get("system_info")
    sys_info = None
    if sys_data:
        sys_info = SystemInfo(
            python_version=sys_data.get("python_version", ""),
            platform=sys_data.get("platform", ""),
            machine=sys_data.get("machine", ""),
            cpu_count=sys_data.get("cpu_count", 0),
            total_memory_gb=sys_data.get("total_memory_gb", 0.0),
            os_version=sys_data.get("os_version", ""),
            timestamp=sys_data.get("timestamp", ""),
            package_versions=sys_data.get("package_versions", {}),
        )

    return BenchmarkSuite(
        name=data.get("name", "loaded"),
        results=results,
        timestamp=data.get("timestamp", ""),
        system_info=sys_info,
    )


# ---------------------------------------------------------------------------
# History tracking
# ---------------------------------------------------------------------------


def append_to_history(
    result: BenchmarkResult,
    history_path: str | Path = "benchmarks/history.jsonl",
) -> Path:
    """Append a benchmark result to the history file (JSONL format).

    Parameters
    ----------
    result : BenchmarkResult
        Result to append.
    history_path : str | Path
        Path to the JSONL history file.

    Returns
    -------
    Path
    """
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    entry = result.to_dict()
    entry["recorded_at"] = datetime.now(timezone.utc).isoformat()

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    return path


def load_history(
    history_path: str | Path = "benchmarks/history.jsonl",
    *,
    name_filter: str | None = None,
) -> pd.DataFrame:
    """Load benchmark history into a DataFrame.

    Parameters
    ----------
    history_path : str | Path
        Path to JSONL history file.
    name_filter : str | None
        Only load entries matching this name.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(history_path)
    if not path.is_file():
        return pd.DataFrame()

    entries: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if name_filter and entry.get("name") != name_filter:
                    continue
                entries.append(entry)
            except json.JSONDecodeError:
                continue

    return pd.DataFrame(entries)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_benchmarks(
    suite: BenchmarkSuite,
    *,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> Any:
    """Create a bar chart of benchmark results.

    Parameters
    ----------
    suite : BenchmarkSuite
        Benchmark suite to visualize.
    output_path : str | Path | None
        If provided, save the figure to this path.
    figsize : tuple[float, float]
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object.
    """
    import matplotlib.pyplot as plt

    df = suite.to_dataframe()
    if df.empty:
        logger.warning("No benchmark data to plot")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Time bar chart
    names = [n.replace("module:", "") for n in df["name"]]
    ax1.barh(names, df["mean_time_s"], xerr=df["std_time_s"], capsize=3, color="#2196F3")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_title("Execution Time")
    ax1.invert_yaxis()

    # Memory bar chart
    ax2.barh(names, df["peak_memory_mb"], color="#4CAF50")
    ax2.set_xlabel("Peak Memory (MB)")
    ax2.set_title("Peak Memory Usage")
    ax2.invert_yaxis()

    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("Saved benchmark plot to %s", output_path)

    return fig


def plot_scalability(
    df: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> Any:
    """Plot scalability analysis results.

    Parameters
    ----------
    df : pd.DataFrame
        Output from :func:`scalability_analysis`.
    output_path : str | Path | None
        If provided, save the figure.
    figsize : tuple[float, float]
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.errorbar(df["n"], df["mean_time_s"], yerr=df["std_time_s"], marker="o", capsize=3, color="#2196F3")
    ax1.set_xlabel("Sample Size (n)")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Execution Time vs. Sample Size")
    ax1.set_xscale("log")

    ax2.plot(df["n"], df["peak_memory_mb"], marker="s", color="#4CAF50")
    ax2.set_xlabel("Sample Size (n)")
    ax2.set_ylabel("Peak Memory (MB)")
    ax2.set_title("Memory vs. Sample Size")
    ax2.set_xscale("log")

    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    suite: BenchmarkSuite,
    *,
    output_path: str | Path | None = None,
) -> str:
    """Generate a Markdown benchmark report.

    Parameters
    ----------
    suite : BenchmarkSuite
        Benchmark suite to report.
    output_path : str | Path | None
        If provided, save the report.

    Returns
    -------
    str
        Markdown report text.
    """
    lines = [
        f"# Benchmark Report: {suite.name}",
        "",
        f"**Date**: {suite.timestamp or datetime.now().isoformat()}",
        "",
    ]

    if suite.system_info:
        si = suite.system_info
        lines.extend(
            [
                "## System Information",
                "",
                f"- **Python**: {si.python_version.split()[0]}",
                f"- **Platform**: {si.os_version}",
                f"- **CPU cores**: {si.cpu_count}",
                f"- **RAM**: {si.total_memory_gb:.1f} GB",
                "",
            ]
        )

    lines.extend(
        [
            "## Results",
            "",
            "| Benchmark | Mean (s) | Std (s) | Min (s) | Max (s) | Memory (MB) |",
            "|-----------|----------|---------|---------|---------|-------------|",
        ]
    )

    for r in suite.results:
        lines.append(
            f"| {r.name} | {r.mean_time:.4f} | {r.std_time:.4f} | "
            f"{r.min_time:.4f} | {r.max_time:.4f} | {r.peak_memory_mb:.1f} |"
        )

    lines.extend(["", "## Statistical Summary", ""])
    for r in suite.results:
        summary = summarize_benchmark(r)
        lines.append(f"### {r.name}")
        lines.append(f"- Iterations: {summary.get('n', 0)}")
        lines.append(f"- CV: {summary.get('cv', 0):.4f}")
        lines.append(f"- Outliers: {summary.get('outliers', 0)}")
        ci = r.ci_95
        lines.append(f"- 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        lines.append("")

    report = "\n".join(lines)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        logger.info("Report saved to %s", path)

    return report


# ---------------------------------------------------------------------------
# Rich rendering
# ---------------------------------------------------------------------------


def render_result(result: BenchmarkResult) -> None:
    """Display a benchmark result with rich formatting.

    Parameters
    ----------
    result : BenchmarkResult
    """
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title=f"Benchmark: {result.name}",
            box=box.ROUNDED,
            header_style="bold cyan",
        )
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Iterations", str(result.iterations))
        table.add_row("Mean", f"{result.mean_time:.4f}s")
        table.add_row("Median", f"{result.median_time:.4f}s")
        table.add_row("Std Dev", f"{result.std_time:.4f}s")
        table.add_row("Min", f"{result.min_time:.4f}s")
        table.add_row("Max", f"{result.max_time:.4f}s")
        ci = result.ci_95
        table.add_row("95% CI", f"[{ci[0]:.4f}, {ci[1]:.4f}]s")
        table.add_row("CV", f"{result.cv:.4f}")
        table.add_row("Peak Memory", f"{result.peak_memory_mb:.1f} MB")

        console.print(table)

    except ImportError:
        print(f"Benchmark: {result.name}")
        print(f"  Mean: {result.mean_time:.4f}s, Std: {result.std_time:.4f}s")
        print(f"  Peak Memory: {result.peak_memory_mb:.1f} MB")


def render_suite(suite: BenchmarkSuite) -> None:
    """Display a benchmark suite with rich formatting.

    Parameters
    ----------
    suite : BenchmarkSuite
    """
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title=f"Benchmark Suite: {suite.name}",
            box=box.ROUNDED,
            header_style="bold cyan",
        )
        table.add_column("Benchmark", style="bold")
        table.add_column("Mean (s)", justify="right")
        table.add_column("Std (s)", justify="right")
        table.add_column("Memory (MB)", justify="right")
        table.add_column("Iters", justify="right")

        for r in suite.results:
            table.add_row(
                r.name,
                f"{r.mean_time:.4f}",
                f"{r.std_time:.4f}",
                f"{r.peak_memory_mb:.1f}",
                str(r.iterations),
            )

        console.print(table)

    except ImportError:
        print(f"Suite: {suite.name}")
        for r in suite.results:
            print(f"  {r.name}: {r.mean_time:.4f}s ({r.peak_memory_mb:.1f} MB)")


def render_regression_report(report: RegressionReport) -> None:
    """Display regression detection results.

    Parameters
    ----------
    report : RegressionReport
    """
    try:
        from rich.console import Console

        console = Console()

        if report.regressions:
            console.print(f"\n[bold red]Performance Regressions Detected ({len(report.regressions)}):[/bold red]")
            for r in report.regressions:
                pct = (r.result_b.mean_time - r.result_a.mean_time) / r.result_a.mean_time * 100
                console.print(f"  [red]{r.name_b}: +{pct:.1f}% slower (p={r.p_value:.4f})[/red]")
        else:
            console.print("\n[green]No performance regressions detected.[/green]")

        if report.improvements:
            console.print(f"\n[bold green]Performance Improvements ({len(report.improvements)}):[/bold green]")
            for r in report.improvements:
                pct = abs((r.result_b.mean_time - r.result_a.mean_time) / r.result_a.mean_time * 100)
                console.print(f"  [green]{r.name_b}: -{pct:.1f}% faster (p={r.p_value:.4f})[/green]")

    except ImportError:
        if report.regressions:
            print(f"REGRESSIONS ({len(report.regressions)}):")
            for r in report.regressions:
                print(f"  {r.summary()}")
        else:
            print("No performance regressions detected.")
