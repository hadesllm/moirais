# SPDX-License-Identifier: AGPL-3.0-or-later

#' Horowitz (1992) smoothed maximum-score estimator
#'
#' @param x Numeric covariate vector or design matrix.
#' @param y Numeric binary response (0/1).
#' @param bandwidth Optional kernel bandwidth (Silverman default).
#' @return Named list with estimate, se, bandwidth, n, method.
#' @keywords internal
hrzb2 <- function(x, y, bandwidth = NULL) {
  y <- as.numeric(y)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  n <- nrow(X); p <- ncol(X)
  if (n < max(10, 2 * p))
    return(list(estimate = rep(NA_real_, p), se = rep(NA_real_, p),
                n = n, method = "smoothed-max-score (insufficient data)"))
  ys <- 2 * y - 1
  h <- if (is.null(bandwidth)) max(.hrz_silverman(X %*% rep(1 / sqrt(p), p)), 1e-3)
       else as.numeric(bandwidth)
  loss <- function(b) {
    nb <- sqrt(sum(b^2)); if (nb < 1e-12) return(1e12)
    bn <- b / nb; z <- (X %*% bn) / h
    -mean(ys * stats::pnorm(z))
  }
  beta0 <- as.numeric(stats::coef(stats::lm.fit(X, ys)))
  nrm <- sqrt(sum(beta0^2)); if (nrm > 1e-12) beta0 <- beta0 / nrm
  if (beta0[1] < 0) beta0 <- -beta0
  res <- stats::optim(beta0, loss, method = "BFGS",
                       control = list(maxit = 200))
  bh <- res$par / max(sqrt(sum(res$par^2)), 1e-12)
  if (bh[1] < 0) bh <- -bh
  z <- (X %*% bh) / h
  phi <- stats::dnorm(z)
  score_i <- -as.numeric(ys * phi) * X / h
  info <- t(score_i) %*% score_i / n
  cov_m <- tryCatch(MASS::ginv(info) / n,
                    error = function(e) matrix(NA, p, p))
  se <- sqrt(pmax(diag(cov_m), 0))
  list(estimate = bh, se = se, bandwidth = h, n = n,
       method = "Horowitz (1992) smoothed maximum-score")
}

# canonical full-name alias (Py<->R API parity)
#' @rdname hrzb2
#' @keywords internal
#' @export
horowitz_smoothed_maximum_score <- hrzb2
