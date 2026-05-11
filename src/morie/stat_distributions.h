/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * stat_distributions.h — MORIE Statistical Distribution Functions
 *
 * Pure C99 implementations of standard probability distributions.
 * No external dependencies beyond math.h.
 *
 * References:
 *   - Abramowitz & Stegun (1964). Handbook of Mathematical Functions.
 *   - Lanczos (1964). A Precision Approximation of the Gamma Function.
 *   - Press et al. (2007). Numerical Recipes, 3rd ed.
 *
 * Compile:
 *   macOS:  cc -O2 -march=native -shared -o stat_distributions.dylib stat_distributions.c -lm
 *   Linux:  cc -O2 -march=native -shared -fPIC -o stat_distributions.so stat_distributions.c -lm
 */

#ifndef MORIE_STAT_DISTRIBUTIONS_H
#define MORIE_STAT_DISTRIBUTIONS_H

#ifdef __cplusplus
extern "C" {
#endif

#define STAT_OK          0
#define STAT_ERR_NULL   -1
#define STAT_ERR_DOMAIN -2
#define STAT_ERR_CONV   -3

double morie_lgamma(double x);
double morie_gamma(double x);
double morie_beta(double a, double b);
double morie_lbeta(double a, double b);

double morie_erf(double x);
double morie_erfc(double x);

double morie_gammainc_lower(double a, double x);
double morie_gammainc_upper(double a, double x);
double morie_gammainc_regularized(double a, double x);

double morie_betainc(double a, double b, double x);

double morie_norm_pdf(double x, double mu, double sigma);
double morie_norm_cdf(double x, double mu, double sigma);
double morie_norm_quantile(double p, double mu, double sigma);

double morie_t_pdf(double x, double df);
double morie_t_cdf(double x, double df);

double morie_chisq_pdf(double x, double df);
double morie_chisq_cdf(double x, double df);

double morie_f_pdf(double x, double df1, double df2);
double morie_f_cdf(double x, double df1, double df2);

double morie_binom_pmf(int k, int n, double p);
double morie_binom_cdf(int k, int n, double p);

double morie_poisson_pmf(int k, double lambda);
double morie_poisson_cdf(int k, double lambda);

#ifdef __cplusplus
}
#endif

#endif /* MORIE_STAT_DISTRIBUTIONS_H */
