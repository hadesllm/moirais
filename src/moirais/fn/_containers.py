"""Shared result dataclasses for moirais.fn functions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# -- Psychometrics --


@dataclass
class RlbRes:
    """Reliability result (Cronbach's alpha)."""

    raw: float
    std: float
    avgr: float
    k: int
    n: int
    ci_lo: float = 0.0
    ci_hi: float = 0.0

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "metric": ["raw", "std", "avgr", "k", "n", "ci_lo", "ci_hi"],
                "value": [self.raw, self.std, self.avgr, self.k, self.n, self.ci_lo, self.ci_hi],
            }
        )


@dataclass
class OmgRes:
    """Omega result (McDonald's omega)."""

    total: float
    hier: float
    alpha: float
    nf: int
    expvar: float = 0.0


@dataclass
class KmoRes:
    """KMO test result."""

    msa: float
    items: dict[str, float] = field(default_factory=dict)


@dataclass
class BrtRes:
    """Bartlett's sphericity result."""

    chisq: float
    df: int
    pval: float


# -- OTIS --


@dataclass
class RplRes:
    """Regional placement result."""

    counts: pd.DataFrame
    props: pd.DataFrame
    year: int
    sex: str | None = None


@dataclass
class AstRes:
    """Alert-state combination result."""

    data: pd.DataFrame
    summary: pd.DataFrame


@dataclass
class VolRes:
    """Volatility result."""

    data: pd.DataFrame
    mean: float
    median: float


@dataclass
class OtDmlR:
    """OTIS DML result."""

    ate: float
    ate_se: float
    ate_pval: float
    att: float
    att_se: float
    att_pval: float
    n: int
    method: str = "IRM"


# -- Effect sizes --


@dataclass
class ESRes:
    """Standardised result for every effect-size calculation."""

    measure: str
    estimate: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    se: float | None = None
    n: int | None = None
    extra: dict = field(default_factory=dict)


# -- Statistical tests --


@dataclass
class TestResult:  # noqa: pytest collection disabled via __test__ = False
    """Result from a hypothesis test."""

    __test__ = False
    test_name: str
    statistic: float
    p_value: float
    df: float | None = None
    method: str = ""
    n: int | None = None
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        sig = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else "ns"
        return f"{self.test_name}: stat={self.statistic:.4f}, p={self.p_value:.4g} {sig}"

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "df": self.df,
            "method": self.method,
            "n": self.n,
            **self.extra,
        }


# -- Regression --


@dataclass
class RegressionResult:
    """Result from a regression model."""

    method: str
    coefficients: dict[str, float]
    se: dict[str, float]
    p_values: dict[str, float]
    r_squared: float | None = None
    adj_r_squared: float | None = None
    residuals: np.ndarray | None = None
    fitted: np.ndarray | None = None
    n: int = 0
    k: int = 0
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"{self.method} (n={self.n}, k={self.k})"]
        if self.r_squared is not None:
            lines.append(f"  R²={self.r_squared:.4f}")
        for name in self.coefficients:
            coef = self.coefficients[name]
            pval = self.p_values.get(name, float("nan"))
            lines.append(f"  {name}: {coef:.4f} (p={pval:.4g})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "coefficients": self.coefficients,
            "se": self.se,
            "p_values": self.p_values,
            "r_squared": self.r_squared,
            "n": self.n,
            **self.extra,
        }


# -- Descriptive --


@dataclass
class DescriptiveResult:
    """Result from a descriptive / exploratory function."""

    name: str
    value: float | dict | pd.DataFrame | None = None
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        return f"{self.name}: {self.value}"

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value, **self.extra}


# -- Time series --


@dataclass
class TimeSeriesResult:
    """Result from a time-series analysis."""

    name: str
    values: np.ndarray | None = None
    lags: np.ndarray | None = None
    ci_upper: np.ndarray | None = None
    ci_lower: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    def __getattr__(self, item: str):
        try:
            return object.__getattribute__(self, "extra")[item]
        except KeyError:
            raise AttributeError(f"'TimeSeriesResult' object has no attribute {item!r}")

    def summary(self) -> str:
        n = len(self.values) if self.values is not None else 0
        return f"{self.name}: {n} values computed"

    def to_dict(self) -> dict:
        return {"name": self.name, **self.extra}


# -- Survival --


@dataclass
class SurvivalResult:
    """Result from a survival analysis."""

    name: str
    times: np.ndarray | None = None
    survival: np.ndarray | None = None
    ci_lower: np.ndarray | None = None
    ci_upper: np.ndarray | None = None
    median_survival: float | None = None
    n_events: int = 0
    n_censored: int = 0
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        return f"{self.name}: {self.n_events} events, {self.n_censored} censored"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "median": self.median_survival,
            "n_events": self.n_events,
            "n_censored": self.n_censored,
            **self.extra,
        }


# -- Diagnostic accuracy --


@dataclass
class DiagnosticResult:
    """Result from a diagnostic accuracy metric."""

    name: str
    estimate: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    n: int = 0
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        ci = f" [{self.ci_lower:.3f}, {self.ci_upper:.3f}]" if self.ci_lower is not None else ""
        return f"{self.name}: {self.estimate:.4f}{ci}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "estimate": self.estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n": self.n,
        }


# -- IRT (Item Response Theory) --


@dataclass
class IRTResult:
    """Result from an IRT model."""

    model: str  # "1PL", "2PL", "3PL", "GRM", etc.
    item_params: dict  # {item: {a, b, c}} or {item: {thresholds}}
    theta: np.ndarray | None = None  # ability estimates
    se_theta: np.ndarray | None = None
    fit: dict = field(default_factory=dict)  # loglik, aic, bic, chi2, p
    info: np.ndarray | None = None  # test information at theta points

    def summary(self) -> str:
        n_items = len(self.item_params)
        return f"IRT {self.model}: {n_items} items"


# -- DIF (Differential Item Functioning) --


@dataclass
class DIFResult:
    """Result from a DIF analysis."""

    method: str  # "MH", "logistic", "Lord", etc.
    items: pd.DataFrame = field(default_factory=pd.DataFrame)  # item, stat, p, effect_size, class
    flagged: list = field(default_factory=list)
    group_var: str = ""

    def summary(self) -> str:
        return f"DIF ({self.method}): {len(self.flagged)} flagged of {len(self.items)} items"


# -- Measurement Invariance --


@dataclass
class InvarianceResult:
    """Result from a measurement invariance test."""

    level: str  # configural, metric, scalar, strict
    fit: dict = field(default_factory=dict)  # cfi, tli, rmsea, srmr, chi2, df, p
    delta_fit: dict = field(default_factory=dict)  # change from previous level
    passed: bool = False

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"MI {self.level}: {status}"


# -- Recidivism --


@dataclass
class RecidivismResult:
    """Result from a recidivism analysis."""

    rate: float
    n_recid: int
    n_total: int
    ci_lower: float = 0.0
    ci_upper: float = 1.0
    subgroup: str | None = None
    method: str = "proportion"

    def summary(self) -> str:
        sub = f" ({self.subgroup})" if self.subgroup else ""
        return f"Recidivism{sub}: {self.rate:.3f} [{self.ci_lower:.3f}, {self.ci_upper:.3f}]"


# -- Custody --


@dataclass
class CustodyResult:
    """Result from a custody/institutional metric."""

    metric: str
    value: float
    n: int = 0
    period: str | None = None
    region: str | None = None
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        loc = f" ({self.region})" if self.region else ""
        return f"{self.metric}{loc}: {self.value:.4f} (n={self.n})"


# -- Multivariate (Maul family) --


@dataclass
class PcaRes:
    """Principal Component Analysis result."""

    components: np.ndarray  # loadings (p x n_components)
    explained_variance: np.ndarray  # eigenvalues
    explained_variance_ratio: np.ndarray
    scores: np.ndarray  # transformed data (n x n_components)
    n: int = 0
    p: int = 0

    def summary(self) -> str:
        cum = float(np.sum(self.explained_variance_ratio))
        return f"PCA: {self.components.shape[1]} components, {cum:.1%} variance explained"


@dataclass
class FaRes:
    """Exploratory Factor Analysis result."""

    loadings: np.ndarray  # (p x n_factors)
    communalities: np.ndarray  # (p,)
    eigenvalues: np.ndarray
    variance_explained: np.ndarray  # per-factor
    n_factors: int = 0
    rotation: str = "varimax"

    def summary(self) -> str:
        return (
            f"EFA: {self.n_factors} factors ({self.rotation}), total var={float(np.sum(self.variance_explained)):.3f}"
        )


@dataclass
class CfaRes:
    """Confirmatory Factor Analysis result."""

    cfi: float
    tli: float
    rmsea: float
    srmr: float
    loadings: dict = field(default_factory=dict)  # {factor: {item: loading}}
    residuals: np.ndarray | None = None

    def summary(self) -> str:
        return f"CFA: CFI={self.cfi:.3f}, TLI={self.tli:.3f}, RMSEA={self.rmsea:.3f}, SRMR={self.srmr:.3f}"


@dataclass
class MdsRes:
    """Multidimensional Scaling result."""

    coordinates: np.ndarray  # (n x n_dims)
    stress: float
    eigenvalues: np.ndarray

    def summary(self) -> str:
        return f"MDS: {self.coordinates.shape[1]}D, stress={self.stress:.4f}"


@dataclass
class TsneRes:
    """t-SNE result."""

    embedding: np.ndarray  # (n x n_dims)

    def summary(self) -> str:
        return f"t-SNE: {self.embedding.shape[0]} points in {self.embedding.shape[1]}D"


@dataclass
class UmapRes:
    """UMAP result."""

    embedding: np.ndarray  # (n x n_dims)

    def summary(self) -> str:
        return f"UMAP: {self.embedding.shape[0]} points in {self.embedding.shape[1]}D"


@dataclass
class KmeansRes:
    """K-means clustering result."""

    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    n_iter: int

    def summary(self) -> str:
        k = self.centers.shape[0]
        return f"K-means: k={k}, inertia={self.inertia:.2f}, iters={self.n_iter}"


@dataclass
class HclstRes:
    """Hierarchical clustering result."""

    labels: np.ndarray
    linkage_matrix: np.ndarray
    distances: np.ndarray | None = None

    def summary(self) -> str:
        k = len(np.unique(self.labels))
        return f"HClust: {k} clusters, {len(self.labels)} observations"


@dataclass
class LdaRes:
    """Linear Discriminant Analysis result."""

    components: np.ndarray  # (p x n_components)
    explained_variance_ratio: np.ndarray
    projected: np.ndarray  # (n x n_components)

    def summary(self) -> str:
        return f"LDA: {self.components.shape[1]} components"


@dataclass
class DbscnRes:
    """DBSCAN clustering result."""

    labels: np.ndarray  # -1 = noise
    n_clusters: int
    n_noise: int

    def summary(self) -> str:
        return f"DBSCAN: {self.n_clusters} clusters, {self.n_noise} noise points"


# -- SIR / Compartmental models --


@dataclass
class SIRResult:
    """Result from a compartmental model (SIR/SEIR/etc)."""

    model: str  # "SIR", "SEIR", etc.
    t: np.ndarray | None = None  # time points
    S: np.ndarray | None = None  # susceptible
    I: np.ndarray | None = None  # infected
    R: np.ndarray | None = None  # recovered
    E: np.ndarray | None = None  # exposed (SEIR only)
    R0: float | None = None
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        peak_I = float(np.max(self.I)) if self.I is not None else 0
        return f"{self.model}: peak infected={peak_I:.4f}, R0={self.R0}"


# -- Spatial statistics --


@dataclass
class SpatialResult:
    """Result from a spatial statistic."""

    name: str
    statistic: float
    p_value: float | None = None
    expected: float | None = None
    variance: float | None = None
    local_values: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        p = f", p={self.p_value:.4g}" if self.p_value is not None else ""
        return f"{self.name}: {self.statistic:.4f}{p}"


# -- Genomics --


@dataclass
class GenomicsResult:
    """Result from a genomics/population genetics test."""

    name: str
    statistic: float
    p_value: float | None = None
    n: int = 0
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        p = f", p={self.p_value:.4g}" if self.p_value is not None else ""
        return f"{self.name}: {self.statistic:.4f}{p}"


# -- Criminology --


@dataclass
class CrimeResult:
    """Result from a criminological metric."""

    name: str
    rate: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    n: int = 0
    population: int = 0
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        ci = f" [{self.ci_lower:.3f}, {self.ci_upper:.3f}]" if self.ci_lower is not None else ""
        return f"{self.name}: {self.rate:.4f}{ci} (n={self.n})"


# -- Signal processing --


@dataclass
class SignalResult:
    """Result from a signal processing function."""

    name: str
    filtered: np.ndarray | None = None
    fs: float = 0.0
    n_samples: int = 0
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        return f"{self.name}: {self.n_samples} samples at {self.fs:.1f} Hz"

    def to_dict(self) -> dict:
        return {"name": self.name, "fs": self.fs, "n_samples": self.n_samples, **self.extra}


# -- Cryptography --


@dataclass
class CryptoResult:
    """Result from a cryptographic operation."""

    algorithm: str
    operation: str
    success: bool = True
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        status = "OK" if self.success else "FAIL"
        return f"{self.algorithm}/{self.operation}: {status}"

    def to_dict(self) -> dict:
        return {"algorithm": self.algorithm, "operation": self.operation, "success": self.success, **self.extra}
