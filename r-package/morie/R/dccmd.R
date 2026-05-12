# SPDX-License-Identifier: GPL-2.0-only

#' DCC multivariate GARCH (Engle 2002)
#'
#' Two-step DCC(1,1) on a panel of return series.
#'
#' @param x Numeric matrix of returns (T x k).
#' @return Named list with \code{a, b, unconditional_correlation,
#'   conditional_correlation, conditional_variance, loglik, n, k, method}.
#' @export
dcc_multivariate_garch <- function(x) {
  X <- as.matrix(x); if (nrow(X) < ncol(X)) X <- t(X)
  n <- nrow(X); k <- ncol(X)
  if (n < 30 || k < 2) stop("Need n>=30, k>=2.")
  if (requireNamespace("rmgarch", quietly = TRUE) &&
      requireNamespace("rugarch", quietly = TRUE)) {
    uspec <- rugarch::multispec(replicate(k, rugarch::ugarchspec(
      variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
      mean.model = list(armaOrder = c(0, 0), include.mean = FALSE)),
      simplify = FALSE))
    dccspec <- rmgarch::dccspec(uspec = uspec, dccOrder = c(1, 1),
                                distribution = "mvnorm")
    fit <- rmgarch::dccfit(dccspec, data = X)
    p <- rmgarch::coef(fit)
    sig_mat <- as.matrix(rmgarch::sigma(fit))
    return(list(a = unname(p["[Joint]dcca1"]),
                b = unname(p["[Joint]dccb1"]),
                unconditional_correlation = cor(X),
                conditional_correlation = rmgarch::rcor(fit),
                conditional_variance = sig_mat^2,
                loglik = as.numeric(rmgarch::likelihood(fit)),
                n = n, k = k,
                method = "DCC(1,1) via rmgarch"))
  }
  # Fallback: two-step EWMA-marginal + closed-form DCC update.
  H <- matrix(NA_real_, n, k); Z <- matrix(NA_real_, n, k)
  for (j in seq_len(k)) {
    rj <- X[, j] - mean(X[, j])
    g <- garch_fit(rj)
    H[, j] <- g$conditional_variance
    Z[, j] <- rj / sqrt(H[, j] + 1e-12)
  }
  Q_bar <- crossprod(Z) / n
  neg_ll <- function(p) {
    a <- p[1]; b <- p[2]
    if (a < 0 || b < 0 || a + b >= 0.9999) return(1e10)
    Q <- Q_bar; ll <- 0
    for (t in seq_len(n)) {
      d <- sqrt(pmax(diag(Q), 1e-12))
      R <- Q / outer(d, d)
      ld <- determinant(R, logarithm = TRUE)
      if (ld$sign <= 0) return(1e10)
      Rinv <- solve(R)
      zt <- Z[t, ]
      ll <- ll + 0.5 * (ld$modulus + sum(zt * (Rinv %*% zt)) - sum(zt^2))
      Q <- (1 - a - b) * Q_bar + a * tcrossprod(zt) + b * Q
    }
    as.numeric(ll)
  }
  opt <- nlminb(c(0.02, 0.95), neg_ll,
                lower = c(1e-6, 1e-6),
                upper = c(0.5, 0.999))
  a <- opt$par[1]; b <- opt$par[2]
  Q <- Q_bar
  R_path <- array(NA_real_, c(n, k, k))
  for (t in seq_len(n)) {
    d <- sqrt(pmax(diag(Q), 1e-12))
    R_path[t, , ] <- Q / outer(d, d)
    Q <- (1 - a - b) * Q_bar + a * tcrossprod(Z[t, ]) + b * Q
  }
  list(a = a, b = b,
       unconditional_correlation = Q_bar,
       conditional_correlation = R_path,
       conditional_variance = H,
       loglik = -opt$objective,
       n = n, k = k,
       method = "DCC(1,1) two-step Gaussian MLE (base R)")
}
