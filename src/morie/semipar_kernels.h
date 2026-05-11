/*
 * semipar_kernels.h — MORIE Semiparametric Kernel Operations
 *
 * Native C kernels for nonparametric and semiparametric estimation:
 * kernel density estimation, Nadaraya-Watson regression, local linear
 * regression, and bandwidth selection. These are the computational
 * primitives used by TMLE, AIPW, and DML nuisance estimators.
 *
 * Security:
 *   - All functions validate inputs (no buffer overflows)
 *   - No heap allocation in kernel evaluation (caller-allocated buffers)
 *   - No global mutable state (thread-safe)
 *
 * References:
 *   - Fan & Gijbels (1996). Local Polynomial Modelling and Its Applications.
 *   - Silverman (1986). Density Estimation for Statistics and Data Analysis.
 *   - Nadaraya (1964). On Estimating Regression. Theory of Prob. and Its App.
 *   - Watson (1964). Smooth Regression Analysis. Sankhya Series A.
 *   - van der Laan & Rose (2011). Targeted Learning. Springer.
 *
 * Compile:
 *   macOS:  cc -O2 -march=native -shared -o semipar_kernels.dylib semipar_kernels.c -lm -framework Accelerate
 *   Linux:  cc -O2 -march=native -shared -fPIC -o semipar_kernels.so semipar_kernels.c -lm
 */

#ifndef MORIE_SEMIPAR_KERNELS_H
#define MORIE_SEMIPAR_KERNELS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes */
#define SEMIPAR_OK          0
#define SEMIPAR_ERR_NULL   -1
#define SEMIPAR_ERR_SIZE   -2
#define SEMIPAR_ERR_BW     -3   /* Invalid bandwidth (zero or negative) */

/* Kernel type enum for KDE */
#define KERNEL_GAUSSIAN      0
#define KERNEL_EPANECHNIKOV  1
#define KERNEL_UNIFORM       2
#define KERNEL_TRIANGULAR    3
#define KERNEL_BIWEIGHT      4

/* --- Kernel Functions ----------------------------------------------------- */

/*
 * K(u) = (2pi)^{-1/2} exp(-u^2/2)
 */
double kernel_gaussian(double u);

/*
 * K(u) = (3/4)(1 - u^2)  for |u| <= 1, else 0
 */
double kernel_epanechnikov(double u);

/*
 * K(u) = 1/2  for |u| <= 1, else 0
 */
double kernel_uniform(double u);

/*
 * K(u) = (1 - |u|)  for |u| <= 1, else 0
 */
double kernel_triangular(double u);

/*
 * K(u) = (15/16)(1 - u^2)^2  for |u| <= 1, else 0
 */
double kernel_biweight(double u);

/* --- Nadaraya-Watson Kernel Regression ------------------------------------ */

/*
 * m_h(x) = sum_i K((x - X_i)/h) Y_i / sum_i K((x - X_i)/h)
 *
 * x:       observed covariate values (n elements)
 * y:       observed outcomes (n elements)
 * n:       number of observations
 * x_eval:  evaluation points (n_eval elements)
 * y_hat:   fitted values at x_eval (n_eval elements, caller-allocated)
 * n_eval:  number of evaluation points
 * bandwidth: kernel bandwidth h > 0
 *
 * Returns: SEMIPAR_OK on success.
 */
int nw_regression(
    const double *x,
    const double *y,
    int           n,
    const double *x_eval,
    double       *y_hat,
    int           n_eval,
    double        bandwidth
);

/* --- Local Linear Regression ---------------------------------------------- */

/*
 * Solves the weighted least squares problem at each evaluation point:
 *   min_{a,b} sum_i K_h(x_eval - X_i) [Y_i - a - b(X_i - x_eval)]^2
 *
 * The fitted value is a (the intercept) and beta_hat is b (the slope).
 * Local linear avoids boundary bias unlike Nadaraya-Watson.
 *
 * x:         observed covariates (n)
 * y:         observed outcomes (n)
 * n:         number of observations
 * x_eval:    evaluation points (n_eval)
 * y_hat:     fitted values at x_eval (n_eval, caller-allocated)
 * beta_hat:  local slopes at x_eval (n_eval, caller-allocated; may be NULL)
 * n_eval:    number of evaluation points
 * bandwidth: kernel bandwidth h > 0
 *
 * Returns: SEMIPAR_OK on success.
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
);

/* --- Kernel Density Estimation -------------------------------------------- */

/*
 * f_h(x) = (1/nh) sum_i K((x - X_i)/h)
 *
 * x:           observed data (n)
 * n:           number of observations
 * x_eval:      evaluation points (n_eval)
 * density:     estimated density at x_eval (n_eval, caller-allocated)
 * n_eval:      number of evaluation points
 * bandwidth:   kernel bandwidth h > 0
 * kernel_type: KERNEL_GAUSSIAN, KERNEL_EPANECHNIKOV, etc.
 *
 * Returns: SEMIPAR_OK on success.
 */
int kde(
    const double *x,
    int           n,
    const double *x_eval,
    double       *density,
    int           n_eval,
    double        bandwidth,
    int           kernel_type
);

/* --- Bandwidth Selection -------------------------------------------------- */

/*
 * Leave-one-out cross-validation for Nadaraya-Watson regression.
 * Minimizes CV(h) = (1/n) sum_i [Y_i - m_{h,-i}(X_i)]^2
 * over a grid of bandwidths from bw_min to bw_max.
 *
 * x:       observed covariates (n)
 * y:       observed outcomes (n)
 * n:       number of observations
 * bw_min:  minimum bandwidth to search
 * bw_max:  maximum bandwidth to search
 * n_grid:  number of grid points (10-100 typical)
 *
 * Returns: optimal bandwidth, or -1.0 on error.
 */
double loocv_bandwidth(
    const double *x,
    const double *y,
    int           n,
    double        bw_min,
    double        bw_max,
    int           n_grid
);

/*
 * Silverman (1986) rule-of-thumb bandwidth:
 *   h = 0.9 * min(sd, IQR/1.34) * n^{-1/5}
 *
 * x: observed data (n)
 * n: number of observations
 *
 * Returns: estimated bandwidth, or -1.0 on error.
 */
double silverman_bandwidth(const double *x, int n);

/* --- Conditional Density Estimation --------------------------------------- */

/*
 * Kernel-weighted mean and variance at each evaluation point.
 * Used by TMLE/AIPW for conditional outcome model smoothing.
 *
 * x:       covariates (n)
 * y:       outcomes (n)
 * n:       number of observations
 * x_eval:  evaluation points (n_eval)
 * mean:    conditional mean at x_eval (n_eval, caller-allocated)
 * var:     conditional variance at x_eval (n_eval, caller-allocated; may be NULL)
 * n_eval:  number of evaluation points
 * bandwidth: kernel bandwidth h > 0
 *
 * Returns: SEMIPAR_OK on success.
 */
int kernel_cond_moments(
    const double *x,
    const double *y,
    int           n,
    const double *x_eval,
    double       *mean,
    double       *var,
    int           n_eval,
    double        bandwidth
);

#ifdef __cplusplus
}
#endif

#endif /* MORIE_SEMIPAR_KERNELS_H */
