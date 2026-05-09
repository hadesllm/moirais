/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * stat_distributions.h — MOIRAIS Statistical Distribution Functions
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

#ifndef MOIRAIS_STAT_DISTRIBUTIONS_H
#define MOIRAIS_STAT_DISTRIBUTIONS_H

#ifdef __cplusplus
extern "C" {
#endif

#define STAT_OK          0
#define STAT_ERR_NULL   -1
#define STAT_ERR_DOMAIN -2
#define STAT_ERR_CONV   -3

double moirais_lgamma(double x);
double moirais_gamma(double x);
double moirais_beta(double a, double b);
double moirais_lbeta(double a, double b);

double moirais_erf(double x);
double moirais_erfc(double x);

double moirais_gammainc_lower(double a, double x);
double moirais_gammainc_upper(double a, double x);
double moirais_gammainc_regularized(double a, double x);

double moirais_betainc(double a, double b, double x);

double moirais_norm_pdf(double x, double mu, double sigma);
double moirais_norm_cdf(double x, double mu, double sigma);
double moirais_norm_quantile(double p, double mu, double sigma);

double moirais_t_pdf(double x, double df);
double moirais_t_cdf(double x, double df);

double moirais_chisq_pdf(double x, double df);
double moirais_chisq_cdf(double x, double df);

double moirais_f_pdf(double x, double df1, double df2);
double moirais_f_cdf(double x, double df1, double df2);

double moirais_binom_pmf(int k, int n, double p);
double moirais_binom_cdf(int k, int n, double p);

double moirais_poisson_pmf(int k, double lambda);
double moirais_poisson_cdf(int k, double lambda);

#ifdef __cplusplus
}
#endif

#endif /* MOIRAIS_STAT_DISTRIBUTIONS_H */
