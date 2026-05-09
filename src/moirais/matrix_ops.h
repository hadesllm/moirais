/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * matrix_ops.h — MOIRAIS Matrix Operations
 *
 * Pure C99 dense linear algebra: LU, Cholesky, QR, eigenvalues, SVD.
 * All matrices are row-major double* with explicit dimensions.
 *
 * References:
 *   - Golub & Van Loan (2013). Matrix Computations, 4th ed.
 *   - Trefethen & Bau (1997). Numerical Linear Algebra. SIAM.
 *
 * Compile:
 *   macOS:  cc -O2 -march=native -shared -o matrix_ops.dylib matrix_ops.c -lm
 *   Linux:  cc -O2 -march=native -shared -fPIC -o matrix_ops.so matrix_ops.c -lm
 */

#ifndef MOIRAIS_MATRIX_OPS_H
#define MOIRAIS_MATRIX_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

#define MAT_OK          0
#define MAT_ERR_NULL   -1
#define MAT_ERR_SIZE   -2
#define MAT_ERR_SING   -3
#define MAT_ERR_CONV   -4
#define MAT_ERR_POSDEF -5

int moirais_mat_mul(
    const double *A, int m, int k,
    const double *B, int bk, int n,
    double *C
);

int moirais_mat_transpose(
    const double *A, int m, int n,
    double *AT
);

int moirais_lu_decompose(
    const double *A, int n,
    double *LU,
    int *piv,
    int *sign
);

int moirais_lu_solve(
    const double *LU, const int *piv, int n,
    const double *b, double *x
);

double moirais_mat_det(const double *A, int n);

int moirais_mat_inverse(const double *A, int n, double *Ainv);

int moirais_cholesky(const double *A, int n, double *L);

int moirais_cholesky_solve(
    const double *L, int n,
    const double *b, double *x
);

int moirais_qr_decompose(
    const double *A, int m, int n,
    double *Q, double *R
);

int moirais_eigen_symmetric(
    const double *A, int n,
    double *eigenvalues,
    double *eigenvectors,
    int max_iter
);

int moirais_svd(
    const double *A, int m, int n,
    double *U, double *S, double *Vt,
    int max_iter
);

#ifdef __cplusplus
}
#endif

#endif /* MOIRAIS_MATRIX_OPS_H */
