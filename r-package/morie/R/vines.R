# SPDX-License-Identifier: AGPL-3.0-or-later
#' Gaussian D-vine pair-copula construction (Aas, Czado, Frigessi & Bakken 2009)
#'
#' Computes the partial-correlation matrix of a multivariate sample
#' (the parameters of a Gaussian D-vine with pair-copulas indexed by
#' partial correlations) and the joint Gaussian copula log-likelihood.
#'
#' @param x matrix (n x d) of continuous variables.
#' @return list: partial_corr, R, loglik, estimate (mean abs off-diag of
#'   partial_corr), n, d, method.
#' @keywords internal
vines <- function(x) {
  x <- as.matrix(x)
  if (nrow(x) < 3L || ncol(x) < 2L)
    return(list(estimate = NA_real_, method = "vine copula (n<3 or d<2)"))
  n <- nrow(x); d <- ncol(x)
  # pseudo-observations
  u <- apply(x, 2, function(z) (rank(z)) / (n + 1))
  z <- stats::qnorm(u)
  R <- stats::cor(z)
  P <- diag(d)
  for (jj in seq_len(d - 1)) {
    for (i in seq_len(d - jj)) {
      if (jj == 1L) {
        P[i, i + jj] <- R[i, i + jj]
      } else {
        cond <- (i + 1):(i + jj - 1)
        idx  <- c(i, i + jj, cond)
        sub  <- R[idx, idx]
        inv  <- MASS::ginv(sub)
        pc   <- -inv[1, 2] / sqrt(inv[1, 1] * inv[2, 2])
        P[i, i + jj] <- pc
      }
      P[i + jj, i] <- P[i, i + jj]
    }
  }
  ld <- determinant(R, logarithm = TRUE)
  if (ld$sign > 0) {
    S <- crossprod(z) / n
    R_inv <- tryCatch(solve(R), error = function(e) MASS::ginv(R))
    loglik <- -0.5 * n * (as.numeric(ld$modulus) + sum(diag(R_inv %*% S)))
  } else {
    loglik <- NA_real_
  }
  list(partial_corr = P, R = R, loglik = as.numeric(loglik),
       estimate = mean(abs(P[upper.tri(P)])),
       n = as.integer(n), d = as.integer(d),
       method = "Gaussian D-vine copula (Aas et al. 2009)")
}

# CANONICAL TEST
# set.seed(0); Sigma <- matrix(c(1, 0.5, 0.3, 0.5, 1, 0.4, 0.3, 0.4, 1), 3)
# z <- MASS::mvrnorm(500, c(0,0,0), Sigma)
# r <- vines(z)
# stopifnot(r$d == 3L, is.finite(r$loglik))

#' @rdname vines
#' @keywords internal
#' @export
vine_copula <- vines
