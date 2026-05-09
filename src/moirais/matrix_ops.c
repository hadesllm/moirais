/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * matrix_ops.c — MOIRAIS Matrix Operations
 *
 * References:
 *   - Golub & Van Loan (2013). Matrix Computations, 4th ed.
 *   - Trefethen & Bau (1997). Numerical Linear Algebra. SIAM.
 */

#include "matrix_ops.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DIM (1 << 14)
#define EPS 1e-14

#define CHECK_NULL(ptr) do { if (!(ptr)) return MAT_ERR_NULL; } while (0)
#define CHECK_SIZE(n)   do { if ((n) <= 0 || (n) > MAX_DIM) return MAT_ERR_SIZE; } while (0)

#define IDX(r, c, cols) ((r) * (cols) + (c))

/* ---- Matrix Multiply ---------------------------------------------------- */

int moirais_mat_mul(
    const double *A, int m, int k,
    const double *B, int bk, int n,
    double *C
) {
    CHECK_NULL(A);
    CHECK_NULL(B);
    CHECK_NULL(C);
    CHECK_SIZE(m);
    CHECK_SIZE(k);
    CHECK_SIZE(n);
    if (bk != k) return MAT_ERR_SIZE;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int p = 0; p < k; p++) {
                sum += A[IDX(i, p, k)] * B[IDX(p, j, n)];
            }
            C[IDX(i, j, n)] = sum;
        }
    }

    return MAT_OK;
}

/* ---- Matrix Transpose --------------------------------------------------- */

int moirais_mat_transpose(
    const double *A, int m, int n,
    double *AT
) {
    CHECK_NULL(A);
    CHECK_NULL(AT);
    CHECK_SIZE(m);
    CHECK_SIZE(n);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            AT[IDX(j, i, m)] = A[IDX(i, j, n)];
        }
    }

    return MAT_OK;
}

/* ---- LU Decomposition (partial pivoting) -------------------------------- */

int moirais_lu_decompose(
    const double *A, int n,
    double *LU,
    int *piv,
    int *sign
) {
    CHECK_NULL(A);
    CHECK_NULL(LU);
    CHECK_NULL(piv);
    CHECK_SIZE(n);

    memcpy(LU, A, n * n * sizeof(double));

    for (int i = 0; i < n; i++) piv[i] = i;
    if (sign) *sign = 1;

    for (int k = 0; k < n; k++) {
        double max_val = 0.0;
        int max_row = k;
        for (int i = k; i < n; i++) {
            double v = fabs(LU[IDX(i, k, n)]);
            if (v > max_val) {
                max_val = v;
                max_row = i;
            }
        }

        if (max_val < EPS) return MAT_ERR_SING;

        if (max_row != k) {
            int tmp = piv[k]; piv[k] = piv[max_row]; piv[max_row] = tmp;
            if (sign) *sign = -(*sign);
            for (int j = 0; j < n; j++) {
                double t = LU[IDX(k, j, n)];
                LU[IDX(k, j, n)] = LU[IDX(max_row, j, n)];
                LU[IDX(max_row, j, n)] = t;
            }
        }

        for (int i = k + 1; i < n; i++) {
            LU[IDX(i, k, n)] /= LU[IDX(k, k, n)];
            for (int j = k + 1; j < n; j++) {
                LU[IDX(i, j, n)] -= LU[IDX(i, k, n)] * LU[IDX(k, j, n)];
            }
        }
    }

    return MAT_OK;
}

/* ---- Solve via LU ------------------------------------------------------- */

int moirais_lu_solve(
    const double *LU, const int *piv, int n,
    const double *b, double *x
) {
    CHECK_NULL(LU);
    CHECK_NULL(piv);
    CHECK_NULL(b);
    CHECK_NULL(x);
    CHECK_SIZE(n);

    for (int i = 0; i < n; i++) x[i] = b[piv[i]];

    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            x[i] -= LU[IDX(i, j, n)] * x[j];
        }
    }

    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) {
            x[i] -= LU[IDX(i, j, n)] * x[j];
        }
        x[i] /= LU[IDX(i, i, n)];
    }

    return MAT_OK;
}

/* ---- Determinant via LU ------------------------------------------------- */

double moirais_mat_det(const double *A, int n) {
    if (!A || n <= 0 || n > MAX_DIM) return 0.0;

    double *LU = (double *)malloc(n * n * sizeof(double));
    int *piv = (int *)malloc(n * sizeof(int));
    if (!LU || !piv) {
        free(LU); free(piv);
        return 0.0;
    }

    int sign;
    int rc = moirais_lu_decompose(A, n, LU, piv, &sign);
    if (rc != MAT_OK) {
        free(LU); free(piv);
        return 0.0;
    }

    double det = (double)sign;
    for (int i = 0; i < n; i++) {
        det *= LU[IDX(i, i, n)];
    }

    free(LU);
    free(piv);
    return det;
}

/* ---- Matrix Inverse via LU ---------------------------------------------- */

int moirais_mat_inverse(const double *A, int n, double *Ainv) {
    CHECK_NULL(A);
    CHECK_NULL(Ainv);
    CHECK_SIZE(n);

    double *LU = (double *)malloc(n * n * sizeof(double));
    int *piv = (int *)malloc(n * sizeof(int));
    double *col = (double *)malloc(n * sizeof(double));
    double *e = (double *)calloc(n, sizeof(double));
    if (!LU || !piv || !col || !e) {
        free(LU); free(piv); free(col); free(e);
        return MAT_ERR_NULL;
    }

    int rc = moirais_lu_decompose(A, n, LU, piv, NULL);
    if (rc != MAT_OK) {
        free(LU); free(piv); free(col); free(e);
        return rc;
    }

    for (int j = 0; j < n; j++) {
        memset(e, 0, n * sizeof(double));
        e[j] = 1.0;
        moirais_lu_solve(LU, piv, n, e, col);
        for (int i = 0; i < n; i++) {
            Ainv[IDX(i, j, n)] = col[i];
        }
    }

    free(LU); free(piv); free(col); free(e);
    return MAT_OK;
}

/* ---- Cholesky Decomposition --------------------------------------------- */

int moirais_cholesky(const double *A, int n, double *L) {
    CHECK_NULL(A);
    CHECK_NULL(L);
    CHECK_SIZE(n);

    memset(L, 0, n * n * sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++) {
                sum += L[IDX(i, k, n)] * L[IDX(j, k, n)];
            }

            if (i == j) {
                double diag = A[IDX(i, i, n)] - sum;
                if (diag <= 0.0) return MAT_ERR_POSDEF;
                L[IDX(i, j, n)] = sqrt(diag);
            } else {
                L[IDX(i, j, n)] = (A[IDX(i, j, n)] - sum) / L[IDX(j, j, n)];
            }
        }
    }

    return MAT_OK;
}

/* ---- Solve SPD System via Cholesky -------------------------------------- */

int moirais_cholesky_solve(
    const double *L, int n,
    const double *b, double *x
) {
    CHECK_NULL(L);
    CHECK_NULL(b);
    CHECK_NULL(x);
    CHECK_SIZE(n);

    double *y = (double *)malloc(n * sizeof(double));
    if (!y) return MAT_ERR_NULL;

    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++) {
            sum += L[IDX(i, j, n)] * y[j];
        }
        y[i] = (b[i] - sum) / L[IDX(i, i, n)];
    }

    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++) {
            sum += L[IDX(j, i, n)] * x[j];
        }
        x[i] = (y[i] - sum) / L[IDX(i, i, n)];
    }

    free(y);
    return MAT_OK;
}

/* ---- QR Decomposition (Householder) ------------------------------------- */

int moirais_qr_decompose(
    const double *A, int m, int n,
    double *Q, double *R
) {
    CHECK_NULL(A);
    CHECK_NULL(Q);
    CHECK_NULL(R);
    CHECK_SIZE(m);
    CHECK_SIZE(n);
    if (m < n) return MAT_ERR_SIZE;

    memcpy(R, A, m * n * sizeof(double));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            Q[IDX(i, j, m)] = (i == j) ? 1.0 : 0.0;
        }
    }

    double *v = (double *)malloc(m * sizeof(double));
    if (!v) return MAT_ERR_NULL;

    int min_mn = (m < n) ? m : n;
    for (int k = 0; k < min_mn; k++) {
        double norm = 0.0;
        for (int i = k; i < m; i++) {
            norm += R[IDX(i, k, n)] * R[IDX(i, k, n)];
        }
        norm = sqrt(norm);

        if (norm < EPS) continue;

        double sign_val = (R[IDX(k, k, n)] >= 0.0) ? 1.0 : -1.0;
        double alpha = -sign_val * norm;

        memset(v, 0, m * sizeof(double));
        for (int i = k; i < m; i++) v[i] = R[IDX(i, k, n)];
        v[k] -= alpha;

        double vnorm = 0.0;
        for (int i = k; i < m; i++) vnorm += v[i] * v[i];
        if (vnorm < EPS) continue;
        double inv_vnorm = 2.0 / vnorm;

        for (int j = k; j < n; j++) {
            double dot = 0.0;
            for (int i = k; i < m; i++) dot += v[i] * R[IDX(i, j, n)];
            for (int i = k; i < m; i++) {
                R[IDX(i, j, n)] -= inv_vnorm * v[i] * dot;
            }
        }

        for (int j = 0; j < m; j++) {
            double dot = 0.0;
            for (int i = k; i < m; i++) dot += v[i] * Q[IDX(i, j, m)];
            for (int i = k; i < m; i++) {
                Q[IDX(i, j, m)] -= inv_vnorm * v[i] * dot;
            }
        }
    }

    double *Qt = (double *)malloc(m * m * sizeof(double));
    if (!Qt) { free(v); return MAT_ERR_NULL; }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            Qt[IDX(i, j, m)] = Q[IDX(j, i, m)];
        }
    }
    memcpy(Q, Qt, m * m * sizeof(double));

    free(v);
    free(Qt);
    return MAT_OK;
}

/* ---- Eigenvalues of Symmetric Matrix (Jacobi iteration) ----------------- */

int moirais_eigen_symmetric(
    const double *A, int n,
    double *eigenvalues,
    double *eigenvectors,
    int max_iter
) {
    CHECK_NULL(A);
    CHECK_NULL(eigenvalues);
    CHECK_SIZE(n);
    if (max_iter <= 0) max_iter = 100;

    double *S = (double *)malloc(n * n * sizeof(double));
    if (!S) return MAT_ERR_NULL;
    memcpy(S, A, n * n * sizeof(double));

    if (eigenvectors) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                eigenvectors[IDX(i, j, n)] = (i == j) ? 1.0 : 0.0;
            }
        }
    }

    for (int iter = 0; iter < max_iter; iter++) {
        double off_diag = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                off_diag += S[IDX(i, j, n)] * S[IDX(i, j, n)];
            }
        }
        if (off_diag < EPS * EPS) break;

        int p = 0, q = 1;
        double max_val = fabs(S[IDX(0, 1, n)]);
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (fabs(S[IDX(i, j, n)]) > max_val) {
                    max_val = fabs(S[IDX(i, j, n)]);
                    p = i;
                    q = j;
                }
            }
        }

        if (max_val < EPS) break;

        double theta;
        double diff = S[IDX(q, q, n)] - S[IDX(p, p, n)];
        if (fabs(diff) < EPS) {
            theta = M_PI / 4.0;
        } else {
            theta = 0.5 * atan2(2.0 * S[IDX(p, q, n)], diff);
        }

        double c = cos(theta);
        double s = sin(theta);

        double *row_p = (double *)malloc(n * sizeof(double));
        double *row_q = (double *)malloc(n * sizeof(double));
        if (!row_p || !row_q) {
            free(S); free(row_p); free(row_q);
            return MAT_ERR_NULL;
        }

        for (int j = 0; j < n; j++) {
            row_p[j] = c * S[IDX(p, j, n)] + s * S[IDX(q, j, n)];
            row_q[j] = -s * S[IDX(p, j, n)] + c * S[IDX(q, j, n)];
        }
        for (int j = 0; j < n; j++) {
            S[IDX(p, j, n)] = row_p[j];
            S[IDX(q, j, n)] = row_q[j];
        }

        for (int i = 0; i < n; i++) {
            double sip = c * S[IDX(i, p, n)] + s * S[IDX(i, q, n)];
            double siq = -s * S[IDX(i, p, n)] + c * S[IDX(i, q, n)];
            S[IDX(i, p, n)] = sip;
            S[IDX(i, q, n)] = siq;
        }

        if (eigenvectors) {
            for (int i = 0; i < n; i++) {
                double vp = c * eigenvectors[IDX(i, p, n)] + s * eigenvectors[IDX(i, q, n)];
                double vq = -s * eigenvectors[IDX(i, p, n)] + c * eigenvectors[IDX(i, q, n)];
                eigenvectors[IDX(i, p, n)] = vp;
                eigenvectors[IDX(i, q, n)] = vq;
            }
        }

        free(row_p);
        free(row_q);
    }

    for (int i = 0; i < n; i++) {
        eigenvalues[i] = S[IDX(i, i, n)];
    }

    free(S);
    return MAT_OK;
}

/* ---- SVD (one-sided Jacobi) --------------------------------------------- */

int moirais_svd(
    const double *A, int m, int n,
    double *U, double *S, double *Vt,
    int max_iter
) {
    CHECK_NULL(A);
    CHECK_NULL(U);
    CHECK_NULL(S);
    CHECK_NULL(Vt);
    CHECK_SIZE(m);
    CHECK_SIZE(n);
    if (m < n) return MAT_ERR_SIZE;
    if (max_iter <= 0) max_iter = 100;

    double *AtA = (double *)malloc(n * n * sizeof(double));
    double *evals = (double *)malloc(n * sizeof(double));
    double *evecs = (double *)malloc(n * n * sizeof(double));
    if (!AtA || !evals || !evecs) {
        free(AtA); free(evals); free(evecs);
        return MAT_ERR_NULL;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += A[IDX(k, i, n)] * A[IDX(k, j, n)];
            }
            AtA[IDX(i, j, n)] = sum;
        }
    }

    int rc = moirais_eigen_symmetric(AtA, n, evals, evecs, max_iter);
    if (rc != MAT_OK) {
        free(AtA); free(evals); free(evecs);
        return rc;
    }

    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (evals[j] > evals[i]) {
                double tmp = evals[i]; evals[i] = evals[j]; evals[j] = tmp;
                for (int k = 0; k < n; k++) {
                    tmp = evecs[IDX(k, i, n)];
                    evecs[IDX(k, i, n)] = evecs[IDX(k, j, n)];
                    evecs[IDX(k, j, n)] = tmp;
                }
            }
        }
    }

    for (int i = 0; i < n; i++) {
        S[i] = (evals[i] > 0.0) ? sqrt(evals[i]) : 0.0;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Vt[IDX(i, j, n)] = evecs[IDX(j, i, n)];
        }
    }

    for (int j = 0; j < n; j++) {
        if (S[j] > EPS) {
            for (int i = 0; i < m; i++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += A[IDX(i, k, n)] * evecs[IDX(k, j, n)];
                }
                U[IDX(i, j, n)] = sum / S[j];
            }
        } else {
            for (int i = 0; i < m; i++) {
                U[IDX(i, j, n)] = 0.0;
            }
        }
    }

    free(AtA); free(evals); free(evecs);
    return MAT_OK;
}
