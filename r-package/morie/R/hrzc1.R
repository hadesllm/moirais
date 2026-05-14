# SPDX-License-Identifier: MIT OR Apache-2.0

#' Powell (1984) censored LAD (CLAD)
#'
#' @param x Numeric covariate vector or design matrix.
#' @param y Numeric response vector.
#' @param censor Censoring threshold (default 0).
#' @return Named list with estimate, se, n, n_uncensored, censor, method.
#' @keywords internal
hrzc1 <- function(x, y, censor = 0.0) {
  y <- as.numeric(y)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  n <- nrow(X); p <- ncol(X); c <- as.numeric(censor)
  if (n < max(10, 2 * p))
    return(list(estimate = rep(NA_real_, p), se = rep(NA_real_, p),
                n = n, method = "CLAD (insufficient data)"))
  keep <- y > c
  if (sum(keep) < max(5, p + 1))
    return(list(estimate = rep(NA_real_, p), se = rep(NA_real_, p),
                n = n, method = "CLAD (too few uncensored obs)"))
  beta <- .hrz_qreg_irls(X[keep, , drop = FALSE], y[keep])
  for (k in 1:30) {
    active <- as.numeric(X %*% beta) > c
    if (sum(active) < max(5, p + 1)) break
    new <- .hrz_qreg_irls(X[active, , drop = FALSE], y[active])
    if (max(abs(new - beta)) < 1e-5) { beta <- new; break }
    beta <- new
  }
  r <- y - as.numeric(X %*% beta); active <- as.numeric(X %*% beta) > c
  if (sum(active) < max(5, p + 1)) {
    se <- rep(NA_real_, p)
  } else {
    Xa <- X[active, , drop = FALSE]; ra <- r[active]
    h <- max(1.06 * stats::sd(ra) * length(ra)^(-1/5), 1e-4)
    f0 <- mean(exp(-0.5 * (ra / h)^2) / (h * sqrt(2 * pi)))
    A <- t(Xa) %*% Xa * f0
    cov_m <- tryCatch(0.25 * MASS::ginv(A) %*% (t(Xa) %*% Xa) %*% MASS::ginv(A),
                      error = function(e) matrix(NA, p, p))
    se <- sqrt(pmax(diag(cov_m), 0))
  }
  list(estimate = if (p == 1) as.numeric(beta) else as.numeric(beta),
       se = if (p == 1) as.numeric(se) else as.numeric(se),
       n = n, n_uncensored = as.integer(sum(active)), censor = c,
       method = "Powell (1984) censored LAD")
}

# canonical full-name alias (Py<->R API parity)
#' @rdname hrzc1
#' @keywords internal
#' @export
horowitz_censored_regression <- hrzc1
