/*
 * semipar_kernels.c — MOIRAIS Semiparametric Kernel Operations
 *
 * Native C99 kernels for nonparametric estimation primitives used in
 * causal inference pipelines (TMLE, AIPW, DML nuisance estimation).
 *
 * Uses Accelerate.framework (vDSP) on macOS where beneficial;
 * portable C fallback on Linux/Windows.
 *
 * Compile:
 *   macOS:  cc -O2 -march=native -shared -o semipar_kernels.dylib semipar_kernels.c -lm -framework Accelerate
 *   Linux:  cc -O2 -march=native -shared -fPIC -o semipar_kernels.so semipar_kernels.c -lm
 *
 * References:
 *   - Fan & Gijbels (1996). Local Polynomial Modelling.
 *   - Silverman (1986). Density Estimation for Statistics and Data Analysis.
 *   - Nadaraya (1964). On Estimating Regression.
 *   - van der Laan & Rose (2011). Targeted Learning. Springer.
 */

#include "semipar_kernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#define HAVE_ACCELERATE 1
#else
#define HAVE_ACCELERATE 0
#endif

#define MAX_OBS (1 << 24)  /* 16M observations */

#define CHECK_NULL(ptr)  do { if (!(ptr)) return SEMIPAR_ERR_NULL; } while (0)
#define CHECK_SIZE(n)    do { if ((n) <= 0 || (n) > MAX_OBS) return SEMIPAR_ERR_SIZE; } while (0)
#define CHECK_BW(h)      do { if ((h) <= 0.0) return SEMIPAR_ERR_BW; } while (0)

static const double INV_SQRT_2PI = 0.3989422804014327;

/* ---- Kernel Functions ---------------------------------------------------- */

double kernel_gaussian(double u) {
    return INV_SQRT_2PI * exp(-0.5 * u * u);
}

double kernel_epanechnikov(double u) {
    if (u < -1.0 || u > 1.0) return 0.0;
    return 0.75 * (1.0 - u * u);
}

double kernel_uniform(double u) {
    if (u < -1.0 || u > 1.0) return 0.0;
    return 0.5;
}

double kernel_triangular(double u) {
    double au = fabs(u);
    if (au > 1.0) return 0.0;
    return 1.0 - au;
}

double kernel_biweight(double u) {
    if (u < -1.0 || u > 1.0) return 0.0;
    double t = 1.0 - u * u;
    return (15.0 / 16.0) * t * t;
}

typedef double (*kernel_fn)(double);

static kernel_fn get_kernel(int kernel_type) {
    switch (kernel_type) {
        case KERNEL_EPANECHNIKOV: return kernel_epanechnikov;
        case KERNEL_UNIFORM:     return kernel_uniform;
        case KERNEL_TRIANGULAR:  return kernel_triangular;
        case KERNEL_BIWEIGHT:    return kernel_biweight;
        default:                 return kernel_gaussian;
    }
}

/* ---- Nadaraya-Watson Regression ------------------------------------------ */

int nw_regression(
    const double *x,
    const double *y,
    int           n,
    const double *x_eval,
    double       *y_hat,
    int           n_eval,
    double        bandwidth
) {
    CHECK_NULL(x);
    CHECK_NULL(y);
    CHECK_NULL(x_eval);
    CHECK_NULL(y_hat);
    CHECK_SIZE(n);
    CHECK_SIZE(n_eval);
    CHECK_BW(bandwidth);

    double inv_h = 1.0 / bandwidth;

    for (int j = 0; j < n_eval; j++) {
        double num = 0.0;
        double den = 0.0;

        for (int i = 0; i < n; i++) {
            double u = (x_eval[j] - x[i]) * inv_h;
            double w = kernel_gaussian(u);
            num += w * y[i];
            den += w;
        }

        /* If no kernel mass, set to global mean of nearby points */
        y_hat[j] = (den > 1e-300) ? num / den : 0.0;
    }

    return SEMIPAR_OK;
}

/* ---- Local Linear Regression --------------------------------------------- */

/*
 * At each evaluation point x0, solve the 2x2 weighted normal equations:
 *
 *   [S0  S1] [a]   [T0]
 *   [S1  S2] [b] = [T1]
 *
 * where Sk = sum_i w_i (X_i - x0)^k, Tk = sum_i w_i Y_i (X_i - x0)^k
 * and w_i = K_h(x0 - X_i).
 *
 * Closed form via Cramer's rule (2x2 determinant):
 *   a = (S2*T0 - S1*T1) / (S0*S2 - S1^2)
 *   b = (S0*T1 - S1*T0) / (S0*S2 - S1^2)
 */
int local_linear(
    const double *x,
    const double *y,
    int           n,
    const double *x_eval,
    double       *y_hat,
    double       *beta_hat,
    int           n_eval,
    double        bandwidth
) {
    CHECK_NULL(x);
    CHECK_NULL(y);
    CHECK_NULL(x_eval);
    CHECK_NULL(y_hat);
    CHECK_SIZE(n);
    CHECK_SIZE(n_eval);
    CHECK_BW(bandwidth);

    double inv_h = 1.0 / bandwidth;

    for (int j = 0; j < n_eval; j++) {
        double s0 = 0.0, s1 = 0.0, s2 = 0.0;
        double t0 = 0.0, t1 = 0.0;
        double x0 = x_eval[j];

        for (int i = 0; i < n; i++) {
            double u = (x0 - x[i]) * inv_h;
            double w = kernel_gaussian(u);
            double dx = x[i] - x0;

            s0 += w;
            s1 += w * dx;
            s2 += w * dx * dx;
            t0 += w * y[i];
            t1 += w * y[i] * dx;
        }

        double det = s0 * s2 - s1 * s1;

        if (fabs(det) < 1e-300) {
            /* Singular — fall back to NW estimate */
            y_hat[j] = (s0 > 1e-300) ? t0 / s0 : 0.0;
            if (beta_hat) beta_hat[j] = 0.0;
        } else {
            double inv_det = 1.0 / det;
            y_hat[j] = (s2 * t0 - s1 * t1) * inv_det;
            if (beta_hat) beta_hat[j] = (s0 * t1 - s1 * t0) * inv_det;
        }
    }

    return SEMIPAR_OK;
}

/* ---- Kernel Density Estimation ------------------------------------------- */

int kde(
    const double *x,
    int           n,
    const double *x_eval,
    double       *density,
    int           n_eval,
    double        bandwidth,
    int           kernel_type
) {
    CHECK_NULL(x);
    CHECK_NULL(x_eval);
    CHECK_NULL(density);
    CHECK_SIZE(n);
    CHECK_SIZE(n_eval);
    CHECK_BW(bandwidth);

    kernel_fn K = get_kernel(kernel_type);
    double inv_h = 1.0 / bandwidth;
    double scale = 1.0 / ((double)n * bandwidth);

    for (int j = 0; j < n_eval; j++) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            double u = (x_eval[j] - x[i]) * inv_h;
            sum += K(u);
        }
        density[j] = sum * scale;
    }

    return SEMIPAR_OK;
}

/* ---- Helper: quickselect for median/IQR --------------------------------- */

static int partition(double *arr, int lo, int hi) {
    double pivot = arr[hi];
    int i = lo;
    for (int j = lo; j < hi; j++) {
        if (arr[j] <= pivot) {
            double tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
            i++;
        }
    }
    double tmp = arr[i]; arr[i] = arr[hi]; arr[hi] = tmp;
    return i;
}

static double quickselect(double *arr, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int p = partition(arr, lo, hi);
        if (p == k) return arr[p];
        else if (p < k) lo = p + 1;
        else hi = p - 1;
    }
    return arr[lo];
}

static double compute_iqr(const double *x, int n) {
    double *buf = (double *)malloc(n * sizeof(double));
    if (!buf) return -1.0;
    memcpy(buf, x, n * sizeof(double));

    int q1_idx = n / 4;
    int q3_idx = (3 * n) / 4;
    double q1 = quickselect(buf, n, q1_idx);

    memcpy(buf, x, n * sizeof(double));
    double q3 = quickselect(buf, n, q3_idx);

    free(buf);
    return q3 - q1;
}

/* ---- Silverman Bandwidth ------------------------------------------------- */

double silverman_bandwidth(const double *x, int n) {
    if (!x || n < 2) return -1.0;

    /* Compute mean */
    double mean = 0.0;
#if HAVE_ACCELERATE
    vDSP_meanvD(x, 1, &mean, n);
#else
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= (double)n;
#endif

    /* Compute standard deviation */
    double var = 0.0;
    for (int i = 0; i < n; i++) {
        double d = x[i] - mean;
        var += d * d;
    }
    double sd = sqrt(var / (double)(n - 1));

    /* IQR */
    double iqr = compute_iqr(x, n);
    if (iqr < 0.0) return -1.0;

    /* h = 0.9 * min(sd, IQR/1.34) * n^{-1/5} */
    double spread = sd;
    double iqr_adj = iqr / 1.34;
    if (iqr_adj > 0.0 && iqr_adj < spread) spread = iqr_adj;

    if (spread <= 0.0) return -1.0;

    return 0.9 * spread * pow((double)n, -0.2);
}

/* ---- LOOCV Bandwidth Selection ------------------------------------------- */

double loocv_bandwidth(
    const double *x,
    const double *y,
    int           n,
    double        bw_min,
    double        bw_max,
    int           n_grid
) {
    if (!x || !y || n < 3 || bw_min <= 0.0 || bw_max <= bw_min || n_grid < 2)
        return -1.0;

    double best_bw = bw_min;
    double best_cv = 1e308;

    double step = (bw_max - bw_min) / (double)(n_grid - 1);

    for (int g = 0; g < n_grid; g++) {
        double h = bw_min + g * step;
        double inv_h = 1.0 / h;
        double cv = 0.0;

        for (int i = 0; i < n; i++) {
            double num = 0.0;
            double den = 0.0;

            for (int j = 0; j < n; j++) {
                if (j == i) continue;  /* leave-one-out */
                double u = (x[i] - x[j]) * inv_h;
                double w = kernel_gaussian(u);
                num += w * y[j];
                den += w;
            }

            double y_hat_i = (den > 1e-300) ? num / den : 0.0;
            double resid = y[i] - y_hat_i;
            cv += resid * resid;
        }

        cv /= (double)n;

        if (cv < best_cv) {
            best_cv = cv;
            best_bw = h;
        }
    }

    return best_bw;
}

/* ---- Conditional Moments ------------------------------------------------- */

int kernel_cond_moments(
    const double *x,
    const double *y,
    int           n,
    const double *x_eval,
    double       *mean,
    double       *var,
    int           n_eval,
    double        bandwidth
) {
    CHECK_NULL(x);
    CHECK_NULL(y);
    CHECK_NULL(x_eval);
    CHECK_NULL(mean);
    CHECK_SIZE(n);
    CHECK_SIZE(n_eval);
    CHECK_BW(bandwidth);

    double inv_h = 1.0 / bandwidth;

    for (int j = 0; j < n_eval; j++) {
        double w_sum = 0.0;
        double wy_sum = 0.0;
        double wy2_sum = 0.0;

        for (int i = 0; i < n; i++) {
            double u = (x_eval[j] - x[i]) * inv_h;
            double w = kernel_gaussian(u);
            w_sum  += w;
            wy_sum += w * y[i];
            wy2_sum += w * y[i] * y[i];
        }

        if (w_sum > 1e-300) {
            mean[j] = wy_sum / w_sum;
            if (var) {
                /* Var = E[Y^2|X] - (E[Y|X])^2 */
                double ey2 = wy2_sum / w_sum;
                double ey  = mean[j];
                double v = ey2 - ey * ey;
                var[j] = (v > 0.0) ? v : 0.0;  /* clamp negative from numerics */
            }
        } else {
            mean[j] = 0.0;
            if (var) var[j] = 0.0;
        }
    }

    return SEMIPAR_OK;
}
