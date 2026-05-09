"""ctypes bridge for MOIRAIS semiparametric C kernels.

Loads ``semipar_kernels.dylib`` (macOS) or ``semipar_kernels.so`` (Linux)
for hardware-accelerated kernel density estimation, Nadaraya-Watson
regression, local linear regression, and bandwidth selection. Falls back
to pure NumPy/SciPy if the shared library is not available.

These primitives underpin the nuisance estimation stage in TMLE, AIPW,
and DML causal inference pipelines.

Compile the C library first::

    # macOS
    cc -O2 -march=native -shared -o semipar_kernels.dylib semipar_kernels.c -lm -framework Accelerate
    # Linux
    cc -O2 -march=native -shared -fPIC -o semipar_kernels.so semipar_kernels.c -lm

Usage
-----
>>> from moirais.semipar_bridge import nw_regression, kde, silverman_bandwidth
>>> import numpy as np
>>> rng = np.random.default_rng(42)
>>> x = rng.standard_normal(200)
>>> y = np.sin(x) + 0.3 * rng.standard_normal(200)
>>> h = silverman_bandwidth(x)
>>> x_grid = np.linspace(-3, 3, 100)
>>> y_hat = nw_regression(x, y, x_grid, h)
>>> density = kde(x, x_grid, h)
"""

from __future__ import annotations

import ctypes
import logging
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

SEMIPAR_OK = 0

_lib: ctypes.CDLL | None = None
_LIB_DIR = Path(__file__).parent

KERNEL_GAUSSIAN = 0
KERNEL_EPANECHNIKOV = 1
KERNEL_UNIFORM = 2
KERNEL_TRIANGULAR = 3
KERNEL_BIWEIGHT = 4


def _load_lib() -> ctypes.CDLL | None:
    """Try to load the semiparametric kernels shared library."""
    global _lib
    if _lib is not None:
        return _lib

    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    lib_path = _LIB_DIR / f"semipar_kernels{suffix}"

    if not lib_path.exists():
        return None

    try:
        lib = ctypes.CDLL(str(lib_path))

        c_double_p = ctypes.POINTER(ctypes.c_double)

        lib.kernel_gaussian.restype = ctypes.c_double
        lib.kernel_gaussian.argtypes = [ctypes.c_double]

        lib.kernel_epanechnikov.restype = ctypes.c_double
        lib.kernel_epanechnikov.argtypes = [ctypes.c_double]

        lib.kernel_uniform.restype = ctypes.c_double
        lib.kernel_uniform.argtypes = [ctypes.c_double]

        lib.kernel_triangular.restype = ctypes.c_double
        lib.kernel_triangular.argtypes = [ctypes.c_double]

        lib.kernel_biweight.restype = ctypes.c_double
        lib.kernel_biweight.argtypes = [ctypes.c_double]

        lib.nw_regression.restype = ctypes.c_int
        lib.nw_regression.argtypes = [
            c_double_p, c_double_p, ctypes.c_int,
            c_double_p, c_double_p, ctypes.c_int,
            ctypes.c_double,
        ]

        lib.local_linear.restype = ctypes.c_int
        lib.local_linear.argtypes = [
            c_double_p, c_double_p, ctypes.c_int,
            c_double_p, c_double_p, c_double_p, ctypes.c_int,
            ctypes.c_double,
        ]

        lib.kde.restype = ctypes.c_int
        lib.kde.argtypes = [
            c_double_p, ctypes.c_int,
            c_double_p, c_double_p, ctypes.c_int,
            ctypes.c_double, ctypes.c_int,
        ]

        lib.loocv_bandwidth.restype = ctypes.c_double
        lib.loocv_bandwidth.argtypes = [
            c_double_p, c_double_p, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_int,
        ]

        lib.silverman_bandwidth.restype = ctypes.c_double
        lib.silverman_bandwidth.argtypes = [c_double_p, ctypes.c_int]

        lib.kernel_cond_moments.restype = ctypes.c_int
        lib.kernel_cond_moments.argtypes = [
            c_double_p, c_double_p, ctypes.c_int,
            c_double_p, c_double_p, c_double_p, ctypes.c_int,
            ctypes.c_double,
        ]

        _lib = lib
        logger.info("Loaded semipar_kernels from %s", lib_path)
        return lib
    except OSError:
        return None


def _as_f64(arr: np.ndarray) -> np.ndarray:
    """Ensure contiguous float64."""
    return np.ascontiguousarray(arr, dtype=np.float64)


def _ptr(arr: np.ndarray):
    """Get ctypes double pointer from numpy array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def is_available() -> bool:
    """Return True if the C kernel library is loaded."""
    return _load_lib() is not None


# ---------------------------------------------------------------------------
# Kernel evaluation
# ---------------------------------------------------------------------------


def kernel_eval(u: float, kernel_type: int = KERNEL_GAUSSIAN) -> float:
    """Evaluate a kernel function at point u.

    Parameters
    ----------
    u : float
        Evaluation point (scaled by bandwidth).
    kernel_type : int
        One of KERNEL_GAUSSIAN (0), KERNEL_EPANECHNIKOV (1),
        KERNEL_UNIFORM (2), KERNEL_TRIANGULAR (3), KERNEL_BIWEIGHT (4).

    Returns
    -------
    float
        Kernel value K(u).
    """
    lib = _load_lib()
    if lib is not None:
        funcs = {
            KERNEL_GAUSSIAN: lib.kernel_gaussian,
            KERNEL_EPANECHNIKOV: lib.kernel_epanechnikov,
            KERNEL_UNIFORM: lib.kernel_uniform,
            KERNEL_TRIANGULAR: lib.kernel_triangular,
            KERNEL_BIWEIGHT: lib.kernel_biweight,
        }
        fn = funcs.get(kernel_type, lib.kernel_gaussian)
        return fn(ctypes.c_double(u))

    if kernel_type == KERNEL_EPANECHNIKOV:
        return 0.75 * (1.0 - u * u) if abs(u) <= 1.0 else 0.0
    elif kernel_type == KERNEL_UNIFORM:
        return 0.5 if abs(u) <= 1.0 else 0.0
    elif kernel_type == KERNEL_TRIANGULAR:
        return (1.0 - abs(u)) if abs(u) <= 1.0 else 0.0
    elif kernel_type == KERNEL_BIWEIGHT:
        if abs(u) > 1.0:
            return 0.0
        t = 1.0 - u * u
        return (15.0 / 16.0) * t * t
    else:
        return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * u * u)


# ---------------------------------------------------------------------------
# Nadaraya-Watson regression
# ---------------------------------------------------------------------------


def nw_regression(
    x: NDArray,
    y: NDArray,
    x_eval: NDArray,
    bandwidth: float,
) -> NDArray:
    r"""Nadaraya-Watson kernel regression.

    .. math::

        \hat{m}_h(x) = \frac{\sum_{i=1}^n K_h(x - X_i) Y_i}
                             {\sum_{i=1}^n K_h(x - X_i)}

    Parameters
    ----------
    x : array, shape (n,)
        Observed covariate values.
    y : array, shape (n,)
        Observed outcomes.
    x_eval : array, shape (n_eval,)
        Points at which to evaluate the regression.
    bandwidth : float
        Kernel bandwidth h > 0.

    Returns
    -------
    y_hat : array, shape (n_eval,)
        Fitted values.

    References
    ----------
    Nadaraya, E. A. (1964). On Estimating Regression. *Theory of Probability
    and Its Applications*, 9(1), 141--142.
    """
    x = _as_f64(np.asarray(x).ravel())
    y = _as_f64(np.asarray(y).ravel())
    x_eval = _as_f64(np.asarray(x_eval).ravel())
    n = len(x)
    n_eval = len(x_eval)
    y_hat = np.empty(n_eval, dtype=np.float64)

    lib = _load_lib()
    if lib is not None:
        rc = lib.nw_regression(
            _ptr(x), _ptr(y), n,
            _ptr(x_eval), _ptr(y_hat), n_eval,
            ctypes.c_double(bandwidth),
        )
        if rc == SEMIPAR_OK:
            return y_hat

    inv_h = 1.0 / bandwidth
    for j in range(n_eval):
        u = (x_eval[j] - x) * inv_h
        w = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * u * u)
        den = w.sum()
        y_hat[j] = (w * y).sum() / den if den > 1e-300 else 0.0

    return y_hat


# ---------------------------------------------------------------------------
# Local linear regression
# ---------------------------------------------------------------------------


def local_linear(
    x: NDArray,
    y: NDArray,
    x_eval: NDArray,
    bandwidth: float,
    return_slope: bool = False,
) -> NDArray | tuple[NDArray, NDArray]:
    r"""Local linear kernel regression (Fan & Gijbels, 1996).

    Avoids boundary bias by fitting a local linear model at each point.

    Parameters
    ----------
    x : array, shape (n,)
        Observed covariate values.
    y : array, shape (n,)
        Observed outcomes.
    x_eval : array, shape (n_eval,)
        Evaluation points.
    bandwidth : float
        Kernel bandwidth h > 0.
    return_slope : bool
        If True, also return the local slope at each evaluation point.

    Returns
    -------
    y_hat : array, shape (n_eval,)
        Fitted values.
    beta_hat : array, shape (n_eval,), optional
        Local slopes (only if return_slope=True).

    References
    ----------
    Fan, J. & Gijbels, I. (1996). *Local Polynomial Modelling and Its
    Applications*. Chapman & Hall.
    """
    x = _as_f64(np.asarray(x).ravel())
    y = _as_f64(np.asarray(y).ravel())
    x_eval = _as_f64(np.asarray(x_eval).ravel())
    n = len(x)
    n_eval = len(x_eval)
    y_hat = np.empty(n_eval, dtype=np.float64)
    beta_hat = np.empty(n_eval, dtype=np.float64) if return_slope else None

    lib = _load_lib()
    if lib is not None:
        beta_ptr = _ptr(beta_hat) if beta_hat is not None else None
        rc = lib.local_linear(
            _ptr(x), _ptr(y), n,
            _ptr(x_eval), _ptr(y_hat), beta_ptr, n_eval,
            ctypes.c_double(bandwidth),
        )
        if rc == SEMIPAR_OK:
            return (y_hat, beta_hat) if return_slope else y_hat

    inv_h = 1.0 / bandwidth
    for j in range(n_eval):
        x0 = x_eval[j]
        u = (x0 - x) * inv_h
        w = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * u * u)
        dx = x - x0
        s0 = w.sum()
        s1 = (w * dx).sum()
        s2 = (w * dx * dx).sum()
        t0 = (w * y).sum()
        t1 = (w * y * dx).sum()
        det = s0 * s2 - s1 * s1
        if abs(det) < 1e-300:
            y_hat[j] = t0 / s0 if s0 > 1e-300 else 0.0
            if beta_hat is not None:
                beta_hat[j] = 0.0
        else:
            inv_det = 1.0 / det
            y_hat[j] = (s2 * t0 - s1 * t1) * inv_det
            if beta_hat is not None:
                beta_hat[j] = (s0 * t1 - s1 * t0) * inv_det

    return (y_hat, beta_hat) if return_slope else y_hat


# ---------------------------------------------------------------------------
# Kernel density estimation
# ---------------------------------------------------------------------------


def kde(
    x: NDArray,
    x_eval: NDArray,
    bandwidth: float,
    kernel_type: int = KERNEL_GAUSSIAN,
) -> NDArray:
    r"""Kernel density estimation.

    .. math::

        \hat{f}_h(x) = \frac{1}{nh} \sum_{i=1}^n K\!\left(\frac{x - X_i}{h}\right)

    Parameters
    ----------
    x : array, shape (n,)
        Observed data.
    x_eval : array, shape (n_eval,)
        Evaluation points.
    bandwidth : float
        Kernel bandwidth h > 0.
    kernel_type : int
        Kernel function (default: KERNEL_GAUSSIAN).

    Returns
    -------
    density : array, shape (n_eval,)
        Estimated density.

    References
    ----------
    Silverman, B. W. (1986). *Density Estimation for Statistics and Data
    Analysis*. Chapman & Hall.
    """
    x = _as_f64(np.asarray(x).ravel())
    x_eval = _as_f64(np.asarray(x_eval).ravel())
    n = len(x)
    n_eval = len(x_eval)
    density = np.empty(n_eval, dtype=np.float64)

    lib = _load_lib()
    if lib is not None:
        rc = lib.kde(
            _ptr(x), n,
            _ptr(x_eval), _ptr(density), n_eval,
            ctypes.c_double(bandwidth), kernel_type,
        )
        if rc == SEMIPAR_OK:
            return density

    _kernel_fns = {
        KERNEL_GAUSSIAN: lambda u: (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * u * u),
        KERNEL_EPANECHNIKOV: lambda u: np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u * u), 0.0),
        KERNEL_UNIFORM: lambda u: np.where(np.abs(u) <= 1.0, 0.5, 0.0),
        KERNEL_TRIANGULAR: lambda u: np.where(np.abs(u) <= 1.0, 1.0 - np.abs(u), 0.0),
        KERNEL_BIWEIGHT: lambda u: np.where(np.abs(u) <= 1.0, (15.0 / 16.0) * (1.0 - u * u) ** 2, 0.0),
    }
    K = _kernel_fns.get(kernel_type, _kernel_fns[KERNEL_GAUSSIAN])

    inv_h = 1.0 / bandwidth
    scale = 1.0 / (n * bandwidth)
    for j in range(n_eval):
        u = (x_eval[j] - x) * inv_h
        density[j] = K(u).sum() * scale

    return density


# ---------------------------------------------------------------------------
# Bandwidth selection
# ---------------------------------------------------------------------------


def silverman_bandwidth(x: NDArray) -> float:
    r"""Silverman rule-of-thumb bandwidth.

    .. math::

        h = 0.9 \cdot \min\!\left(\hat\sigma,\; \frac{\mathrm{IQR}}{1.34}\right)
            \cdot n^{-1/5}

    Parameters
    ----------
    x : array, shape (n,)
        Observed data.

    Returns
    -------
    float
        Estimated bandwidth.

    References
    ----------
    Silverman, B. W. (1986). *Density Estimation for Statistics and Data
    Analysis*, p. 48. Chapman & Hall.
    """
    x = _as_f64(np.asarray(x).ravel())
    n = len(x)

    lib = _load_lib()
    if lib is not None:
        result = lib.silverman_bandwidth(_ptr(x), n)
        if result > 0.0:
            return result

    sd = np.std(x, ddof=1)
    iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
    spread = min(sd, iqr / 1.34) if iqr > 0.0 else sd
    if spread <= 0.0:
        return 1.0
    return 0.9 * spread * n ** (-0.2)


def loocv_bandwidth(
    x: NDArray,
    y: NDArray,
    bw_min: float | None = None,
    bw_max: float | None = None,
    n_grid: int = 30,
) -> float:
    r"""Leave-one-out cross-validation bandwidth for NW regression.

    Minimizes:

    .. math::

        \mathrm{CV}(h) = \frac{1}{n}\sum_{i=1}^n
            \left[Y_i - \hat{m}_{h,-i}(X_i)\right]^2

    Parameters
    ----------
    x : array, shape (n,)
        Observed covariates.
    y : array, shape (n,)
        Observed outcomes.
    bw_min : float, optional
        Minimum bandwidth (default: 0.1 * silverman).
    bw_max : float, optional
        Maximum bandwidth (default: 2.0 * silverman).
    n_grid : int
        Number of grid points to search.

    Returns
    -------
    float
        Optimal bandwidth.

    References
    ----------
    Hardle, W. (1990). *Applied Nonparametric Regression*. Cambridge.
    """
    x = _as_f64(np.asarray(x).ravel())
    y = _as_f64(np.asarray(y).ravel())
    n = len(x)

    h_rot = silverman_bandwidth(x)
    if bw_min is None:
        bw_min = 0.1 * h_rot
    if bw_max is None:
        bw_max = 2.0 * h_rot
    if bw_min <= 0.0:
        bw_min = 0.01
    if bw_max <= bw_min:
        bw_max = bw_min + 1.0

    lib = _load_lib()
    if lib is not None:
        result = lib.loocv_bandwidth(
            _ptr(x), _ptr(y), n,
            ctypes.c_double(bw_min), ctypes.c_double(bw_max), n_grid,
        )
        if result > 0.0:
            return result

    best_bw = bw_min
    best_cv = np.inf
    grid = np.linspace(bw_min, bw_max, n_grid)

    for h in grid:
        inv_h = 1.0 / h
        cv = 0.0
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            u = (x[i] - x[mask]) * inv_h
            w = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * u * u)
            den = w.sum()
            y_hat_i = (w * y[mask]).sum() / den if den > 1e-300 else 0.0
            cv += (y[i] - y_hat_i) ** 2
        cv /= n
        if cv < best_cv:
            best_cv = cv
            best_bw = h

    return best_bw


# ---------------------------------------------------------------------------
# Conditional moments
# ---------------------------------------------------------------------------


def kernel_cond_moments(
    x: NDArray,
    y: NDArray,
    x_eval: NDArray,
    bandwidth: float,
    return_variance: bool = True,
) -> NDArray | tuple[NDArray, NDArray]:
    r"""Kernel-weighted conditional mean and variance.

    Used by TMLE/AIPW for smoothing the conditional outcome model.

    .. math::

        \hat{E}[Y \mid X=x] = \frac{\sum_i K_h(x - X_i)\, Y_i}
                                     {\sum_i K_h(x - X_i)}

    Parameters
    ----------
    x : array, shape (n,)
        Observed covariates.
    y : array, shape (n,)
        Observed outcomes.
    x_eval : array, shape (n_eval,)
        Evaluation points.
    bandwidth : float
        Kernel bandwidth h > 0.
    return_variance : bool
        If True, also return conditional variance.

    Returns
    -------
    mean : array, shape (n_eval,)
        Conditional mean.
    var : array, shape (n_eval,), optional
        Conditional variance (if return_variance=True).
    """
    x = _as_f64(np.asarray(x).ravel())
    y = _as_f64(np.asarray(y).ravel())
    x_eval = _as_f64(np.asarray(x_eval).ravel())
    n = len(x)
    n_eval = len(x_eval)
    mean_out = np.empty(n_eval, dtype=np.float64)
    var_out = np.empty(n_eval, dtype=np.float64) if return_variance else None

    lib = _load_lib()
    if lib is not None:
        var_ptr = _ptr(var_out) if var_out is not None else None
        rc = lib.kernel_cond_moments(
            _ptr(x), _ptr(y), n,
            _ptr(x_eval), _ptr(mean_out), var_ptr, n_eval,
            ctypes.c_double(bandwidth),
        )
        if rc == SEMIPAR_OK:
            return (mean_out, var_out) if return_variance else mean_out

    inv_h = 1.0 / bandwidth
    for j in range(n_eval):
        u = (x_eval[j] - x) * inv_h
        w = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * u * u)
        w_sum = w.sum()
        if w_sum > 1e-300:
            mean_out[j] = (w * y).sum() / w_sum
            if var_out is not None:
                ey2 = (w * y * y).sum() / w_sum
                v = ey2 - mean_out[j] ** 2
                var_out[j] = max(v, 0.0)
        else:
            mean_out[j] = 0.0
            if var_out is not None:
                var_out[j] = 0.0

    return (mean_out, var_out) if return_variance else mean_out


# ---------------------------------------------------------------------------
# Compile helper
# ---------------------------------------------------------------------------


class SemiparKernels:
    """Object-oriented interface to semiparametric C kernels.

    Wraps the module-level functions and reports which backend is active.

    Examples
    --------
    >>> from moirais.semipar_bridge import SemiparKernels
    >>> sk = SemiparKernels()
    >>> print(sk.backend)  # 'c' or 'numpy'
    >>> bw = sk.silverman_bandwidth(x)
    >>> y_hat = sk.nw_regression(x, y, x_eval, bw)
    """

    def __init__(self) -> None:
        self._c_available = _load_lib() is not None
        if self._c_available:
            logger.info("SemiparKernels using C backend")
        else:
            logger.info("SemiparKernels using NumPy backend")

    @property
    def available(self) -> bool:
        """True if the C library is loaded and ready."""
        return self._c_available

    @property
    def backend(self) -> str:
        """Return 'c' or 'numpy' depending on which backend is active."""
        return "c" if self._c_available else "numpy"

    def nw_regression(
        self,
        x: NDArray,
        y: NDArray,
        x_eval: NDArray,
        bandwidth: float,
    ) -> NDArray:
        r"""Nadaraya-Watson kernel regression. See :func:`nw_regression`."""
        return nw_regression(x, y, x_eval, bandwidth)

    def local_linear(
        self,
        x: NDArray,
        y: NDArray,
        x_eval: NDArray,
        bandwidth: float,
    ) -> tuple[NDArray, NDArray]:
        r"""Local linear regression. See :func:`local_linear`."""
        return local_linear(x, y, x_eval, bandwidth, return_slope=True)

    def kde(
        self,
        x: NDArray,
        x_eval: NDArray,
        bandwidth: float,
        kernel: str = "gaussian",
    ) -> NDArray:
        r"""Kernel density estimation. See :func:`kde`."""
        _name_map = {
            "gaussian": KERNEL_GAUSSIAN,
            "epanechnikov": KERNEL_EPANECHNIKOV,
            "uniform": KERNEL_UNIFORM,
            "triangular": KERNEL_TRIANGULAR,
            "biweight": KERNEL_BIWEIGHT,
        }
        ktype = _name_map.get(kernel.lower())
        if ktype is None:
            raise ValueError(
                f"Unknown kernel '{kernel}'. "
                f"Choose from: {', '.join(sorted(_name_map))}"
            )
        return kde(x, x_eval, bandwidth, kernel_type=ktype)

    def loocv_bandwidth(
        self,
        x: NDArray,
        y: NDArray,
        bw_min: float | None = None,
        bw_max: float | None = None,
        n_grid: int = 30,
    ) -> float:
        r"""LOOCV bandwidth selection. See :func:`loocv_bandwidth`."""
        return loocv_bandwidth(x, y, bw_min, bw_max, n_grid)

    def silverman_bandwidth(self, x: NDArray) -> float:
        r"""Silverman rule-of-thumb bandwidth. See :func:`silverman_bandwidth`."""
        return silverman_bandwidth(x)


# ---------------------------------------------------------------------------
# Compile helper
# ---------------------------------------------------------------------------


def compile_semipar_lib(
    output_dir: str | Path | None = None,
    optimize: bool = True,
) -> Path | None:
    """Attempt to compile the semiparametric kernels C library.

    Parameters
    ----------
    output_dir : str or Path, optional
        Where to write the shared library. Defaults to the moirais package dir.
    optimize : bool
        Use ``-O2 -march=native`` (default True).

    Returns
    -------
    Path or None
        Path to compiled library, or None on failure.
    """
    import shutil
    import subprocess

    cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if cc is None:
        logger.warning("No C compiler found")
        return None

    pkg_dir = Path(__file__).parent
    src = pkg_dir / "semipar_kernels.c"
    header = pkg_dir / "semipar_kernels.h"
    if not src.exists() or not header.exists():
        logger.warning("semipar_kernels.c or .h not found at %s", pkg_dir)
        return None

    if output_dir is None:
        output_dir = pkg_dir
    output_dir = Path(output_dir)

    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = output_dir / f"semipar_kernels{suffix}"

    cmd = [cc]
    if optimize:
        cmd += ["-O2", "-march=native"]
    cmd += ["-shared"]
    if sys.platform != "darwin":
        cmd.append("-fPIC")
    cmd += ["-o", str(output), str(src), "-lm"]
    if sys.platform == "darwin":
        cmd += ["-framework", "Accelerate"]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Compiled semipar_kernels to %s", output)
        return output
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("Compilation failed: %s", e)
        return None
