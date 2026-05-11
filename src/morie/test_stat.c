/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * test_stat.c — MORIE C Library Test Suite
 *
 * Compile & run:
 *   cc -O2 -o test_stat test_stat.c stat_distributions.c signal_processing.c matrix_ops.c -lm
 *   ./test_stat
 */

#include "stat_distributions.h"
#include "signal_processing.h"
#include "matrix_ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT_NEAR(a, b, tol, msg) do { \
    tests_run++; \
    double _a = (a), _b = (b); \
    if (fabs(_a - _b) < (tol)) { \
        tests_passed++; \
    } else { \
        printf("FAIL: %s: got %.10g, expected %.10g (diff %.2e)\n", \
               msg, _a, _b, fabs(_a - _b)); \
    } \
} while (0)

#define ASSERT_TRUE(cond, msg) do { \
    tests_run++; \
    if (cond) { \
        tests_passed++; \
    } else { \
        printf("FAIL: %s\n", msg); \
    } \
} while (0)

/* ---- Distribution Tests ------------------------------------------------- */

static void test_gamma(void) {
    ASSERT_NEAR(morie_gamma(1.0), 1.0, 1e-10, "gamma(1)");
    ASSERT_NEAR(morie_gamma(2.0), 1.0, 1e-10, "gamma(2)");
    ASSERT_NEAR(morie_gamma(3.0), 2.0, 1e-10, "gamma(3)");
    ASSERT_NEAR(morie_gamma(4.0), 6.0, 1e-10, "gamma(4)");
    ASSERT_NEAR(morie_gamma(5.0), 24.0, 1e-8, "gamma(5)");
    ASSERT_NEAR(morie_gamma(0.5), sqrt(M_PI), 1e-10, "gamma(0.5)");
}

static void test_beta(void) {
    ASSERT_NEAR(morie_beta(1.0, 1.0), 1.0, 1e-10, "beta(1,1)");
    ASSERT_NEAR(morie_beta(2.0, 2.0), 1.0 / 6.0, 1e-10, "beta(2,2)");
    ASSERT_NEAR(morie_beta(0.5, 0.5), M_PI, 1e-10, "beta(0.5,0.5)");
}

static void test_erf(void) {
    ASSERT_NEAR(morie_erf(0.0), 0.0, 1e-10, "erf(0)");
    ASSERT_NEAR(morie_erf(1.0), 0.8427007929, 1e-6, "erf(1)");
    ASSERT_NEAR(morie_erf(-1.0), -0.8427007929, 1e-6, "erf(-1)");
    ASSERT_NEAR(morie_erfc(0.0), 1.0, 1e-10, "erfc(0)");
}

static void test_norm(void) {
    ASSERT_NEAR(morie_norm_pdf(0.0, 0.0, 1.0), 0.3989422804, 1e-8, "norm_pdf(0)");
    ASSERT_NEAR(morie_norm_cdf(0.0, 0.0, 1.0), 0.5, 1e-8, "norm_cdf(0)");
    ASSERT_NEAR(morie_norm_cdf(1.96, 0.0, 1.0), 0.975002, 1e-4, "norm_cdf(1.96)");
    ASSERT_NEAR(morie_norm_cdf(-1.96, 0.0, 1.0), 0.024998, 1e-4, "norm_cdf(-1.96)");
    ASSERT_NEAR(morie_norm_quantile(0.5, 0.0, 1.0), 0.0, 1e-6, "norm_quantile(0.5)");
    ASSERT_NEAR(morie_norm_quantile(0.975, 0.0, 1.0), 1.96, 1e-3, "norm_quantile(0.975)");
    ASSERT_NEAR(morie_norm_quantile(0.025, 0.0, 1.0), -1.96, 1e-3, "norm_quantile(0.025)");
}

static void test_t(void) {
    ASSERT_NEAR(morie_t_cdf(0.0, 10.0), 0.5, 1e-8, "t_cdf(0, df=10)");
    ASSERT_TRUE(morie_t_cdf(2.228, 10.0) > 0.97, "t_cdf(2.228, df=10) > 0.97");
    ASSERT_TRUE(morie_t_cdf(2.228, 10.0) < 0.98, "t_cdf(2.228, df=10) < 0.98");
    ASSERT_TRUE(morie_t_pdf(0.0, 10.0) > 0.0, "t_pdf(0, df=10) > 0");
}

static void test_chisq(void) {
    ASSERT_NEAR(morie_chisq_cdf(0.0, 5.0), 0.0, 1e-8, "chisq_cdf(0, df=5)");
    ASSERT_TRUE(morie_chisq_cdf(11.07, 5.0) > 0.94, "chisq_cdf(11.07, df=5) > 0.94");
    ASSERT_TRUE(morie_chisq_pdf(5.0, 5.0) > 0.0, "chisq_pdf(5, df=5) > 0");
}

static void test_f(void) {
    ASSERT_NEAR(morie_f_cdf(0.0, 5.0, 10.0), 0.0, 1e-8, "f_cdf(0)");
    ASSERT_TRUE(morie_f_cdf(3.33, 5.0, 10.0) > 0.94, "f_cdf(3.33, 5, 10) > 0.94");
    ASSERT_TRUE(morie_f_pdf(1.0, 5.0, 10.0) > 0.0, "f_pdf(1) > 0");
}

static void test_binom(void) {
    ASSERT_NEAR(morie_binom_pmf(0, 10, 0.5), 0.000976563, 1e-6, "binom_pmf(0,10,0.5)");
    ASSERT_NEAR(morie_binom_pmf(5, 10, 0.5), 0.24609375, 1e-6, "binom_pmf(5,10,0.5)");
    ASSERT_NEAR(morie_binom_cdf(10, 10, 0.5), 1.0, 1e-8, "binom_cdf(10,10,0.5)");
    ASSERT_TRUE(morie_binom_cdf(5, 10, 0.5) > 0.6, "binom_cdf(5,10,0.5) > 0.6");
}

static void test_poisson(void) {
    ASSERT_NEAR(morie_poisson_pmf(0, 5.0), exp(-5.0), 1e-10, "poisson_pmf(0,5)");
    ASSERT_NEAR(morie_poisson_pmf(5, 5.0), pow(5.0, 5) * exp(-5.0) / 120.0, 1e-8, "poisson_pmf(5,5)");
    ASSERT_TRUE(morie_poisson_cdf(10, 5.0) > 0.95, "poisson_cdf(10,5) > 0.95");
}

static void test_betainc(void) {
    ASSERT_NEAR(morie_betainc(1.0, 1.0, 0.5), 0.5, 1e-8, "betainc(1,1,0.5)");
    ASSERT_NEAR(morie_betainc(1.0, 1.0, 0.0), 0.0, 1e-8, "betainc(1,1,0)");
    ASSERT_NEAR(morie_betainc(1.0, 1.0, 1.0), 1.0, 1e-8, "betainc(1,1,1)");
    ASSERT_NEAR(morie_betainc(2.0, 2.0, 0.5), 0.5, 1e-6, "betainc(2,2,0.5)");
}

static void test_gammainc(void) {
    ASSERT_NEAR(morie_gammainc_regularized(1.0, 1.0), 1.0 - exp(-1.0), 1e-8, "gammainc(1,1)");
    ASSERT_TRUE(morie_gammainc_regularized(5.0, 10.0) > 0.9, "gammainc(5,10) > 0.9");
}

/* ---- Signal Processing Tests -------------------------------------------- */

static void test_fft_roundtrip(void) {
    int n = 8;
    float re[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float im[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float orig[8];
    memcpy(orig, re, sizeof(orig));

    int rc = morie_fft(re, im, n);
    ASSERT_TRUE(rc == DSP_OK, "fft returns ok");

    ASSERT_NEAR(re[0], 36.0f, 1e-4, "fft dc component");

    rc = morie_ifft(re, im, n);
    ASSERT_TRUE(rc == DSP_OK, "ifft returns ok");

    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(re[i], orig[i], 1e-4, "fft roundtrip");
    }
}

static void test_fft_invalid(void) {
    float re[3] = {1, 2, 3};
    float im[3] = {0, 0, 0};
    ASSERT_TRUE(morie_fft(re, im, 3) == DSP_ERR_SIZE, "fft non-pow2 rejected");
    ASSERT_TRUE(morie_fft(NULL, im, 4) == DSP_ERR_NULL, "fft null rejected");
}

static void test_windows(void) {
    int n = 64;
    float w[64];

    morie_window_hamming(w, n);
    ASSERT_NEAR(w[0], 0.08f, 0.01f, "hamming start");
    ASSERT_TRUE(w[n / 2] > 0.9f, "hamming center > 0.9");

    morie_window_hanning(w, n);
    ASSERT_NEAR(w[0], 0.0f, 1e-5, "hanning start = 0");
    ASSERT_NEAR(w[n / 2], 1.0f, 0.02f, "hanning center ~ 1");

    morie_window_blackman(w, n);
    ASSERT_NEAR(w[0], 0.0f, 0.01f, "blackman start ~ 0");
    ASSERT_TRUE(w[n / 2] > 0.9f, "blackman center > 0.9");
}

static void test_convolve(void) {
    float x[] = {1.0f, 2.0f, 3.0f};
    float h[] = {1.0f, 1.0f};
    float out[4];

    int rc = morie_convolve(x, 3, h, 2, out, 4);
    ASSERT_TRUE(rc == DSP_OK, "convolve ok");
    ASSERT_NEAR(out[0], 1.0f, 1e-5, "conv[0]");
    ASSERT_NEAR(out[1], 3.0f, 1e-5, "conv[1]");
    ASSERT_NEAR(out[2], 5.0f, 1e-5, "conv[2]");
    ASSERT_NEAR(out[3], 3.0f, 1e-5, "conv[3]");
}

static void test_convolve_fft(void) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h[] = {0.5f, 0.5f};
    float out_td[5];
    float out_fft[5];

    morie_convolve(x, 4, h, 2, out_td, 5);
    morie_convolve_fft(x, 4, h, 2, out_fft, 5);

    for (int i = 0; i < 5; i++) {
        ASSERT_NEAR(out_fft[i], out_td[i], 1e-4, "fft conv matches td conv");
    }
}

static void test_autocorrelation(void) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float r[4];

    int rc = morie_autocorrelation(x, 8, r, 4);
    ASSERT_TRUE(rc == DSP_OK, "autocorr ok");
    ASSERT_NEAR(r[0], 1.0f, 1e-5, "autocorr lag 0 = 1");
    ASSERT_TRUE(r[1] > 0.0f, "autocorr lag 1 > 0 for trend");
}

static void test_fir_filter(void) {
    float x[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float b[] = {0.25f, 0.5f, 0.25f};
    float out[5];

    int rc = morie_fir_filter(x, 5, b, 3, out);
    ASSERT_TRUE(rc == DSP_OK, "fir ok");
    ASSERT_NEAR(out[0], 0.25f, 1e-5, "fir impulse[0]");
    ASSERT_NEAR(out[1], 0.5f, 1e-5, "fir impulse[1]");
    ASSERT_NEAR(out[2], 0.25f, 1e-5, "fir impulse[2]");
    ASSERT_NEAR(out[3], 0.0f, 1e-5, "fir impulse[3]");
}

static void test_iir_filter(void) {
    float x[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float b[] = {1.0f};
    float a[] = {1.0f, -0.5f};
    float out[8];

    int rc = morie_iir_filter(x, 8, b, 1, a, 2, out);
    ASSERT_TRUE(rc == DSP_OK, "iir ok");
    ASSERT_NEAR(out[0], 1.0f, 1e-5, "iir[0]");
    ASSERT_NEAR(out[1], 0.5f, 1e-5, "iir[1]");
    ASSERT_NEAR(out[2], 0.25f, 1e-5, "iir[2]");
}

static void test_zcr_rms(void) {
    float sine[64];
    for (int i = 0; i < 64; i++) {
        sine[i] = sinf(2.0f * (float)M_PI * (float)i / 16.0f);
    }

    float zcr = morie_zero_crossing_rate(sine, 64);
    ASSERT_TRUE(zcr > 0.1f, "zcr sine > 0.1");
    ASSERT_TRUE(zcr < 0.5f, "zcr sine < 0.5");

    float rms = morie_rms_energy(sine, 64);
    ASSERT_NEAR(rms, 1.0f / sqrtf(2.0f), 0.05f, "rms sine ~ 1/sqrt(2)");
}

/* ---- Matrix Tests ------------------------------------------------------- */

static void test_mat_mul(void) {
    double A[] = {1, 2, 3, 4, 5, 6};
    double B[] = {7, 8, 9, 10, 11, 12};
    double C[4];

    int rc = morie_mat_mul(A, 2, 3, B, 3, 2, C);
    ASSERT_TRUE(rc == MAT_OK, "matmul ok");
    ASSERT_NEAR(C[0], 58.0, 1e-10, "matmul[0,0]");
    ASSERT_NEAR(C[1], 64.0, 1e-10, "matmul[0,1]");
    ASSERT_NEAR(C[2], 139.0, 1e-10, "matmul[1,0]");
    ASSERT_NEAR(C[3], 154.0, 1e-10, "matmul[1,1]");
}

static void test_mat_transpose(void) {
    double A[] = {1, 2, 3, 4, 5, 6};
    double AT[6];

    int rc = morie_mat_transpose(A, 2, 3, AT);
    ASSERT_TRUE(rc == MAT_OK, "transpose ok");
    ASSERT_NEAR(AT[0], 1.0, 1e-10, "transpose[0]");
    ASSERT_NEAR(AT[1], 4.0, 1e-10, "transpose[1]");
    ASSERT_NEAR(AT[2], 2.0, 1e-10, "transpose[2]");
    ASSERT_NEAR(AT[3], 5.0, 1e-10, "transpose[3]");
}

static void test_lu_solve(void) {
    double A[] = {2, 1, -1, -3, -1, 2, -2, 1, 2};
    double b[] = {8, -11, -3};
    double x[3];
    double LU[9];
    int piv[3];

    int rc = morie_lu_decompose(A, 3, LU, piv, NULL);
    ASSERT_TRUE(rc == MAT_OK, "lu decompose ok");

    rc = morie_lu_solve(LU, piv, 3, b, x);
    ASSERT_TRUE(rc == MAT_OK, "lu solve ok");
    ASSERT_NEAR(x[0], 2.0, 1e-8, "lu solve x[0]");
    ASSERT_NEAR(x[1], 3.0, 1e-8, "lu solve x[1]");
    ASSERT_NEAR(x[2], -1.0, 1e-8, "lu solve x[2]");
}

static void test_det(void) {
    double I3[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    ASSERT_NEAR(morie_mat_det(I3, 3), 1.0, 1e-10, "det(I3)");

    double A[] = {1, 2, 3, 4};
    ASSERT_NEAR(morie_mat_det(A, 2), -2.0, 1e-10, "det 2x2");
}

static void test_inverse(void) {
    double A[] = {4, 7, 2, 6};
    double Ainv[4];
    double I2[4];

    int rc = morie_mat_inverse(A, 2, Ainv);
    ASSERT_TRUE(rc == MAT_OK, "inverse ok");

    morie_mat_mul(A, 2, 2, Ainv, 2, 2, I2);
    ASSERT_NEAR(I2[0], 1.0, 1e-8, "A*Ainv[0,0]");
    ASSERT_NEAR(I2[1], 0.0, 1e-8, "A*Ainv[0,1]");
    ASSERT_NEAR(I2[2], 0.0, 1e-8, "A*Ainv[1,0]");
    ASSERT_NEAR(I2[3], 1.0, 1e-8, "A*Ainv[1,1]");
}

static void test_cholesky(void) {
    double A[] = {4, 2, 2, 3};
    double L[4];

    int rc = morie_cholesky(A, 2, L);
    ASSERT_TRUE(rc == MAT_OK, "cholesky ok");
    ASSERT_NEAR(L[0], 2.0, 1e-10, "chol L[0,0]");
    ASSERT_NEAR(L[1], 0.0, 1e-10, "chol L[0,1]");
    ASSERT_NEAR(L[2], 1.0, 1e-10, "chol L[1,0]");
    ASSERT_NEAR(L[3], sqrt(2.0), 1e-10, "chol L[1,1]");
}

static void test_cholesky_solve(void) {
    double A[] = {4, 2, 2, 3};
    double L[4];
    double b[] = {1, 2};
    double x[2];

    morie_cholesky(A, 2, L);
    int rc = morie_cholesky_solve(L, 2, b, x);
    ASSERT_TRUE(rc == MAT_OK, "chol solve ok");

    double Ax[2];
    morie_mat_mul(A, 2, 2, x, 2, 1, Ax);
    ASSERT_NEAR(Ax[0], b[0], 1e-8, "chol solve Ax[0]=b[0]");
    ASSERT_NEAR(Ax[1], b[1], 1e-8, "chol solve Ax[1]=b[1]");
}

static void test_qr(void) {
    double A[] = {1, -1, 4, 2, -2, 1, 2, 1, 6};
    double Q[9], R[9];

    int rc = morie_qr_decompose(A, 3, 3, Q, R);
    ASSERT_TRUE(rc == MAT_OK, "qr ok");

    double QR[9];
    morie_mat_mul(Q, 3, 3, R, 3, 3, QR);
    for (int i = 0; i < 9; i++) {
        ASSERT_NEAR(QR[i], A[i], 1e-8, "QR = A");
    }

    double QtQ[9];
    double Qt[9];
    morie_mat_transpose(Q, 3, 3, Qt);
    morie_mat_mul(Qt, 3, 3, Q, 3, 3, QtQ);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            ASSERT_NEAR(QtQ[i * 3 + j], expected, 1e-8, "Q orthogonal");
        }
    }
}

static void test_eigen_symmetric(void) {
    double A[] = {2, 1, 1, 2};
    double evals[2];
    double evecs[4];

    int rc = morie_eigen_symmetric(A, 2, evals, evecs, 100);
    ASSERT_TRUE(rc == MAT_OK, "eigen ok");

    double e1 = (evals[0] > evals[1]) ? evals[0] : evals[1];
    double e2 = (evals[0] > evals[1]) ? evals[1] : evals[0];
    ASSERT_NEAR(e1, 3.0, 1e-6, "eigenvalue 1");
    ASSERT_NEAR(e2, 1.0, 1e-6, "eigenvalue 2");
}

static void test_svd(void) {
    double A[] = {3, 2, 2, 2, 3, -2};
    double U[4], S[2], Vt[4];

    int rc = morie_svd(A, 3, 2, U, S, Vt, 200);
    ASSERT_TRUE(rc == MAT_OK, "svd ok");
    ASSERT_TRUE(S[0] >= S[1], "singular values sorted");
    ASSERT_TRUE(S[0] > 0.0, "s[0] > 0");
    ASSERT_TRUE(S[1] > 0.0, "s[1] > 0");

    double USVt[6];
    double US[6];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            US[i * 2 + j] = U[i * 2 + j] * S[j];
        }
    }
    morie_mat_mul(US, 3, 2, Vt, 2, 2, USVt);
    for (int i = 0; i < 6; i++) {
        ASSERT_NEAR(USVt[i], A[i], 1e-6, "USVt = A");
    }
}

static void test_singular_matrix(void) {
    double A[] = {1, 2, 2, 4};
    double LU[4];
    int piv[2];
    int rc = morie_lu_decompose(A, 2, LU, piv, NULL);
    ASSERT_TRUE(rc == MAT_ERR_SING, "singular matrix detected");
}

static void test_not_posdef(void) {
    double A[] = {1, 2, 2, 1};
    double L[4];
    int rc = morie_cholesky(A, 2, L);
    ASSERT_TRUE(rc == MAT_ERR_POSDEF, "non-posdef detected");
}

static void test_null_checks(void) {
    ASSERT_TRUE(morie_mat_mul(NULL, 2, 2, NULL, 2, 2, NULL) == MAT_ERR_NULL, "matmul null");
    ASSERT_TRUE(morie_fft(NULL, NULL, 4) == DSP_ERR_NULL, "fft null");
    ASSERT_TRUE(morie_window_hamming(NULL, 4) == DSP_ERR_NULL, "window null");
}

/* ---- Main --------------------------------------------------------------- */

int main(void) {
    printf("MORIE C Library Test Suite\n");
    printf("========================\n\n");

    printf("[stat_distributions]\n");
    test_gamma();
    test_beta();
    test_erf();
    test_norm();
    test_t();
    test_chisq();
    test_f();
    test_binom();
    test_poisson();
    test_betainc();
    test_gammainc();

    printf("[signal_processing]\n");
    test_fft_roundtrip();
    test_fft_invalid();
    test_windows();
    test_convolve();
    test_convolve_fft();
    test_autocorrelation();
    test_fir_filter();
    test_iir_filter();
    test_zcr_rms();

    printf("[matrix_ops]\n");
    test_mat_mul();
    test_mat_transpose();
    test_lu_solve();
    test_det();
    test_inverse();
    test_cholesky();
    test_cholesky_solve();
    test_qr();
    test_eigen_symmetric();
    test_svd();
    test_singular_matrix();
    test_not_posdef();
    test_null_checks();

    printf("\n========================\n");
    printf("%d/%d tests passed\n", tests_passed, tests_run);

    return (tests_passed == tests_run) ? 0 : 1;
}
