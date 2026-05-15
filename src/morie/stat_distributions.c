/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * stat_distributions.c — MORIE Statistical Distribution Functions
 *
 * References:
 *   - Abramowitz & Stegun (1964). Handbook of Mathematical Functions.
 *   - Lanczos (1964). A Precision Approximation of the Gamma Function.
 *   - Press et al. (2007). Numerical Recipes, 3rd ed.
 */

#include "stat_distributions.h"
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

#define LANCZOS_G 7.0
#define MAX_ITER 500
#define EPS 1e-14

static const double lanczos_coeff[9] = {
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
};

/* ---- Gamma / Log-Gamma (Lanczos) ---------------------------------------- */

double morie_lgamma(double x) {
    if (x <= 0.0) return 1e308;

    if (x < 0.5) {
        return log(M_PI / sin(M_PI * x)) - morie_lgamma(1.0 - x);
    }

    x -= 1.0;
    double a = lanczos_coeff[0];
    double t = x + LANCZOS_G + 0.5;

    for (int i = 1; i < 9; i++) {
        a += lanczos_coeff[i] / (x + (double)i);
    }

    return 0.5 * log(2.0 * M_PI) + (x + 0.5) * log(t) - t + log(a);
}

double morie_gamma(double x) {
    return exp(morie_lgamma(x));
}

double morie_lbeta(double a, double b) {
    return morie_lgamma(a) + morie_lgamma(b) - morie_lgamma(a + b);
}

double morie_beta(double a, double b) {
    return exp(morie_lbeta(a, b));
}

/* ---- Error Function (Abramowitz & Stegun 7.1.26) ------------------------ */

double morie_erf(double x) {
    if (x == 0.0) return 0.0;

    int sign = 1;
    if (x < 0.0) {
        sign = -1;
        x = -x;
    }

    double t = 1.0 / (1.0 + 0.3275911 * x);
    double t2 = t * t;
    double t3 = t2 * t;
    double t4 = t3 * t;
    double t5 = t4 * t;

    double y = 1.0 - (0.254829592 * t
                      - 0.284496736 * t2
                      + 1.421413741 * t3
                      - 1.453152027 * t4
                      + 1.061405429 * t5) * exp(-x * x);

    return sign * y;
}

double morie_erfc(double x) {
    return 1.0 - morie_erf(x);
}

/* ---- Incomplete Gamma (series + continued fraction) --------------------- */

static double gammainc_series(double a, double x) {
    double term = 1.0 / a;
    double sum = term;

    for (int n = 1; n < MAX_ITER; n++) {
        term *= x / (a + (double)n);
        sum += term;
        if (fabs(term) < fabs(sum) * EPS) break;
    }

    return sum * exp(-x + a * log(x) - morie_lgamma(a));
}

static double gammainc_cf(double a, double x) {
    double b0 = x + 1.0 - a;
    double f = (fabs(b0) < 1e-30) ? 1e-30 : b0;
    double c = f;
    double d = 1.0 / f;
    double result = d;

    for (int n = 1; n < MAX_ITER; n++) {
        double an = (double)n * (a - (double)n);
        double bn = x + 2.0 * (double)n + 1.0 - a;
        d = bn + an * d;
        if (fabs(d) < 1e-30) d = 1e-30;
        c = bn + an / c;
        if (fabs(c) < 1e-30) c = 1e-30;
        d = 1.0 / d;
        double delta = d * c;
        result *= delta;
        if (fabs(delta - 1.0) < EPS) break;
    }

    return 1.0 - result * exp(-x + a * log(x) - morie_lgamma(a));
}

double morie_gammainc_regularized(double a, double x) {
    if (x < 0.0 || a <= 0.0) return 0.0;
    if (x == 0.0) return 0.0;

    if (x < a + 1.0) {
        return gammainc_series(a, x);
    }
    return gammainc_cf(a, x);
}

double morie_gammainc_lower(double a, double x) {
    return morie_gammainc_regularized(a, x) * morie_gamma(a);
}

double morie_gammainc_upper(double a, double x) {
    return (1.0 - morie_gammainc_regularized(a, x)) * morie_gamma(a);
}

/* ---- Regularized Incomplete Beta (continued fraction) ------------------- */

static double betainc_cf(double a, double b, double x) {
    double qab = a + b;
    double qap = a + 1.0;
    double qam = a - 1.0;

    double c = 1.0;
    double d = 1.0 / (1.0 - qab * x / qap);
    if (fabs(d) < 1e-30) d = 1e-30;
    double f = d;

    for (int m = 1; m < MAX_ITER; m++) {
        double m2 = 2.0 * (double)m;

        double aa = (double)m * (b - (double)m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (fabs(d) < 1e-30) d = 1e-30;
        c = 1.0 + aa / c;
        if (fabs(c) < 1e-30) c = 1e-30;
        d = 1.0 / d;
        f *= d * c;

        aa = -(a + (double)m) * (qab + (double)m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (fabs(d) < 1e-30) d = 1e-30;
        c = 1.0 + aa / c;
        if (fabs(c) < 1e-30) c = 1e-30;
        d = 1.0 / d;
        double delta = d * c;
        f *= delta;

        if (fabs(delta - 1.0) < EPS) break;
    }

    return f;
}

double morie_betainc(double a, double b, double x) {
    if (x < 0.0 || x > 1.0) return 0.0;
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;

    double front = exp(morie_lgamma(a + b) - morie_lgamma(a) - morie_lgamma(b)
                       + a * log(x) + b * log(1.0 - x));

    if (x < (a + 1.0) / (a + b + 2.0)) {
        return front * betainc_cf(a, b, x) / a;
    }
    return 1.0 - front * betainc_cf(b, a, 1.0 - x) / b;
}

/* ---- Normal Distribution ------------------------------------------------ */

double morie_norm_pdf(double x, double mu, double sigma) {
    if (sigma <= 0.0) return 0.0;
    double z = (x - mu) / sigma;
    return exp(-0.5 * z * z) / (sigma * sqrt(2.0 * M_PI));
}

double morie_norm_cdf(double x, double mu, double sigma) {
    if (sigma <= 0.0) return 0.0;
    double z = (x - mu) / sigma;
    return 0.5 * (1.0 + morie_erf(z / M_SQRT2));
}

static double norm_cdf_std(double x) {
    return 0.5 * (1.0 + morie_erf(x / M_SQRT2));
}

double morie_norm_quantile(double p, double mu, double sigma) {
    if (p <= 0.0 || p >= 1.0 || sigma <= 0.0) return 0.0;

    double t;
    if (p < 0.5) {
        t = sqrt(-2.0 * log(p));
    } else {
        t = sqrt(-2.0 * log(1.0 - p));
    }

    double c0 = 2.515517;
    double c1 = 0.802853;
    double c2 = 0.010328;
    double d1 = 1.432788;
    double d2 = 0.189269;
    double d3 = 0.001308;

    double z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if (p < 0.5) z = -z;

    for (int i = 0; i < 4; i++) {
        double phi = norm_cdf_std(z);
        double pdf = exp(-0.5 * z * z) / sqrt(2.0 * M_PI);
        if (pdf < 1e-300) break;
        z -= (phi - p) / pdf;
    }

    return mu + sigma * z;
}

/* ---- t-Distribution ----------------------------------------------------- */

double morie_t_pdf(double x, double df) {
    if (df <= 0.0) return 0.0;
    double half_df = 0.5 * df;
    double coeff = exp(morie_lgamma(half_df + 0.5) - morie_lgamma(half_df))
                   / sqrt(df * M_PI);
    return coeff * pow(1.0 + x * x / df, -(half_df + 0.5));
}

double morie_t_cdf(double x, double df) {
    if (df <= 0.0) return 0.0;
    double t2 = x * x;
    double v = df / (df + t2);
    double ib = morie_betainc(0.5 * df, 0.5, v);
    if (x >= 0.0) {
        return 0.5 + 0.5 * (1.0 - ib);
    }
    return 0.5 * ib;
}

/* ---- Chi-Squared Distribution ------------------------------------------- */

double morie_chisq_pdf(double x, double df) {
    if (x < 0.0 || df <= 0.0) return 0.0;
    if (x == 0.0) {
        if (df == 2.0) return 0.5;
        if (df < 2.0) return 1e308;
        return 0.0;
    }
    double half_df = 0.5 * df;
    return exp((half_df - 1.0) * log(x) - 0.5 * x - half_df * log(2.0) - morie_lgamma(half_df));
}

double morie_chisq_cdf(double x, double df) {
    if (x < 0.0 || df <= 0.0) return 0.0;
    return morie_gammainc_regularized(0.5 * df, 0.5 * x);
}

/* ---- F-Distribution ----------------------------------------------------- */

double morie_f_pdf(double x, double df1, double df2) {
    if (x < 0.0 || df1 <= 0.0 || df2 <= 0.0) return 0.0;
    if (x == 0.0) {
        if (df1 == 2.0) return 1.0;
        if (df1 < 2.0) return 1e308;
        return 0.0;
    }
    double half1 = 0.5 * df1;
    double half2 = 0.5 * df2;
    return exp(half1 * log(df1) + half2 * log(df2)
               + (half1 - 1.0) * log(x)
               - (half1 + half2) * log(df1 * x + df2)
               - morie_lbeta(half1, half2));
}

double morie_f_cdf(double x, double df1, double df2) {
    if (x < 0.0 || df1 <= 0.0 || df2 <= 0.0) return 0.0;
    double v = df1 * x / (df1 * x + df2);
    return morie_betainc(0.5 * df1, 0.5 * df2, v);
}

/* ---- Binomial Distribution ---------------------------------------------- */

double morie_binom_pmf(int k, int n, double p) {
    if (k < 0 || k > n || n < 0 || p < 0.0 || p > 1.0) return 0.0;
    if (p == 0.0) return (k == 0) ? 1.0 : 0.0;
    if (p == 1.0) return (k == n) ? 1.0 : 0.0;

    double log_coeff = morie_lgamma((double)(n + 1))
                       - morie_lgamma((double)(k + 1))
                       - morie_lgamma((double)(n - k + 1));
    return exp(log_coeff + (double)k * log(p) + (double)(n - k) * log(1.0 - p));
}

double morie_binom_cdf(int k, int n, double p) {
    if (k < 0 || n < 0 || p < 0.0 || p > 1.0) return 0.0;
    if (k >= n) return 1.0;
    return morie_betainc((double)(n - k), (double)(k + 1), 1.0 - p);
}

/* ---- Poisson Distribution ----------------------------------------------- */

double morie_poisson_pmf(int k, double lambda) {
    if (k < 0 || lambda < 0.0) return 0.0;
    if (lambda == 0.0) return (k == 0) ? 1.0 : 0.0;
    return exp((double)k * log(lambda) - lambda - morie_lgamma((double)(k + 1)));
}

double morie_poisson_cdf(int k, double lambda) {
    if (k < 0 || lambda < 0.0) return 0.0;
    if (lambda == 0.0) return 1.0;
    return 1.0 - morie_gammainc_regularized((double)(k + 1), lambda);
}
