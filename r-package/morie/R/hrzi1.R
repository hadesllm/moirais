# SPDX-License-Identifier: AGPL-3.0-or-later

#' Ichimura (1993) single-index model
#'
#' @param x Numeric covariate vector or design matrix.
#' @param y Numeric response vector.
#' @param bandwidth Optional kernel bandwidth (Silverman default).
#' @return Named list with estimate, se, bandwidth, n, loss, method.
#' @keywords internal
hrzi1 <- function(x, y, bandwidth = NULL) {
  y <- as.numeric(y)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  n <- nrow(X); p <- ncol(X)
  if (n < max(10, 2 * p))
    return(list(estimate = rep(NA_real_, p), se = rep(NA_real_, p),
                n = n, method = "single-index (insufficient data)"))
  beta0 <- as.numeric(stats::coef(stats::lm.fit(X, y)))
  nrm <- sqrt(sum(beta0^2))
  if (nrm < 1e-10) beta0 <- rep(1, p) / sqrt(p) else beta0 <- beta0 / nrm
  if (beta0[1] < 0) beta0 <- -beta0
  h0 <- if (is.null(bandwidth)) .hrz_silverman(X %*% beta0) else as.numeric(bandwidth)

  obj <- function(b) {
    nb <- sqrt(sum(b^2)); if (nb < 1e-12) return(1e12)
    bn <- b / nb; idx <- as.numeric(X %*% bn)
    u <- outer(idx, idx, `-`) / h0
    w <- exp(-0.5 * u^2); diag(w) <- 0
    wsum <- rowSums(w); safe <- ifelse(wsum > 0, wsum, 1)
    g_hat <- as.numeric((w %*% y) / safe)
    mean((y - g_hat)^2)
  }
  res <- stats::optim(beta0, obj, method = "Nelder-Mead",
                       control = list(maxit = 200, reltol = 1e-5))
  bh <- res$par; bh <- bh / max(sqrt(sum(bh^2)), 1e-12)
  if (bh[1] < 0) bh <- -bh
  # Numerical Hessian for SE
  eps <- 1e-4; H <- matrix(0, p, p)
  for (i in 1:p) for (j in 1:p) {
    bp <- bh; bp[i] <- bp[i] + eps; bp[j] <- bp[j] + eps
    bm <- bh; bm[i] <- bm[i] - eps; bm[j] <- bm[j] - eps
    bpm <- bh; bpm[i] <- bpm[i] + eps; bpm[j] <- bpm[j] - eps
    bmp <- bh; bmp[i] <- bmp[i] - eps; bmp[j] <- bmp[j] + eps
    H[i, j] <- (obj(bp) - obj(bpm) - obj(bmp) + obj(bm)) / (4 * eps^2)
  }
  H <- 0.5 * (H + t(H))
  cov_m <- tryCatch(MASS::ginv(H) / n, error = function(e) matrix(NA, p, p))
  se <- sqrt(pmax(diag(cov_m), 0))
  list(estimate = bh, se = se, bandwidth = h0, n = n, loss = res$value,
       method = "Ichimura (1993) single-index model")
}

# canonical full-name alias (Py<->R API parity)
#' @rdname hrzi1
#' @keywords internal
#' @export
horowitz_index_model <- hrzi1
