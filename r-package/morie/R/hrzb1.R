# SPDX-License-Identifier: GPL-2.0-only

#' Manski (1975) maximum-score estimator
#'
#' @param x Numeric covariate vector or design matrix.
#' @param y Numeric binary response (0/1).
#' @return Named list with estimate, se, score, n, method.
#' @keywords internal
hrzb1 <- function(x, y) {
  y <- as.numeric(y)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  n <- nrow(X); p <- ncol(X)
  if (n < max(10, 2 * p))
    return(list(estimate = rep(NA_real_, p), se = rep(NA_real_, p),
                n = n, method = "maximum-score (insufficient data)"))
  ys <- 2 * y - 1
  score <- function(b) -mean(ys * (X %*% b > 0))
  beta0 <- as.numeric(stats::coef(stats::lm.fit(X, ys)))
  nrm <- sqrt(sum(beta0^2)); if (nrm > 1e-12) beta0 <- beta0 / nrm
  if (beta0[1] < 0) beta0 <- -beta0
  best <- beta0; best_l <- score(best)
  set.seed(0)
  for (k in 1:8) {
    s <- stats::rnorm(p); s <- s / sqrt(sum(s^2))
    r <- stats::optim(s, score, method = "Nelder-Mead",
                       control = list(maxit = 300, reltol = 1e-4))
    b <- r$par / max(sqrt(sum(r$par^2)), 1e-12)
    if (b[1] < 0) b <- -b
    l <- score(b)
    if (l < best_l) { best_l <- l; best <- b }
  }
  # Subsample SE (cube-root rescale)
  set.seed(42); B <- 30; m <- max(20L, n %/% 2L)
  boot <- matrix(0, B, p)
  for (b_idx in 1:B) {
    idx <- sample.int(n, m, replace = FALSE)
    Xb <- X[idx, , drop = FALSE]; yb <- ys[idx]
    sc <- function(b) -mean(yb * (Xb %*% b > 0))
    r <- stats::optim(best + 0.05 * stats::rnorm(p), sc, method = "Nelder-Mead",
                       control = list(maxit = 150))
    bb <- r$par / max(sqrt(sum(r$par^2)), 1e-12)
    if (bb[1] < 0) bb <- -bb
    boot[b_idx, ] <- bb
  }
  se <- apply(boot, 2, stats::sd) * (m / n)^(1/3)
  list(estimate = best, se = se, score = -best_l, n = n,
       method = "Manski (1975) maximum-score (binary response)",
       warnings = list("Cube-root asymptotics: subsample-rescaled SEs."))
}

# canonical full-name alias (Py<->R API parity)
#' @rdname hrzb1
#' @keywords internal
#' @export
horowitz_binary_response <- hrzb1
