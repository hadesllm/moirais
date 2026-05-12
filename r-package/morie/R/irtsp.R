# SPDX-License-Identifier: GPL-2.0-only

#' IRT 2PL spatial ideal-point model (Armstrong Ch 4)
#'
#' Joint-MLE alternating updates for Clinton-Jackman-Rivers 2PL.
#' Standardises latent x to mean 0, SD 1 every sweep for identification.
#'
#' @param x Binary roll-call matrix (n by m).
#' @param n_iter Outer sweeps (default 60).
#' @param tol Convergence tolerance.
#' @return Named list with `x_hat`, `alpha`, `beta`, `loglik`, `n_iter`,
#'   `method`.
#' @export
irtsp <- function(x, n_iter = 60L, tol = 1e-6) {
  logistic <- function(z) 1 / (1 + exp(-pmin(pmax(z, -30), 30)))
  M <- if (is.matrix(x)) x else matrix(as.numeric(x), ncol = 1L)
  n <- nrow(M); m <- ncol(M)
  if (n < 2L || m < 1L)
    return(list(x_hat = rep(NA_real_, n), alpha = rep(NA_real_, m),
                beta = rep(NA_real_, m), loglik = NA_real_,
                n_iter = 0L, method = "irt_spatial"))
  Mc <- M - matrix(colMeans(M, na.rm = TRUE), n, m, byrow = TRUE)
  Mc[is.na(Mc)] <- 0
  sv <- tryCatch(svd(Mc, nu = 1L, nv = 0L),
                 error = function(e) NULL)
  x_hat <- if (!is.null(sv)) sv$u[, 1] * sv$d[1]
           else seq(-1, 1, length.out = n)
  x_hat <- (x_hat - mean(x_hat)) / (stats::sd(x_hat) + 1e-12)
  alpha <- rep(1, m); beta <- rep(0, m)
  prev_ll <- -Inf
  it <- 0L
  for (it in seq_len(n_iter)) {
    for (j in seq_len(m)) {
      yj <- M[, j]; mask <- !is.na(yj)
      xj <- x_hat[mask]; yjm <- yj[mask]
      a <- alpha[j]; b <- beta[j]
      for (k in 1:5) {
        z <- a * (xj - b); p <- logistic(z); w <- p * (1 - p) + 1e-9
        ga <- sum((yjm - p) * (xj - b))
        gb <- sum((yjm - p) * (-a))
        Haa <- -sum(w * (xj - b)^2)
        Hbb <- -sum(w * a * a)
        Hab <- -sum(w * (-a * (xj - b) + (yjm - p) * (-1)))
        H <- matrix(c(Haa, Hab, Hab, Hbb), 2, 2)
        g <- c(ga, gb)
        step <- tryCatch(solve(H - 1e-6 * diag(2), g),
                         error = function(e) NULL)
        if (is.null(step)) break
        a <- a - step[1]; b <- b - step[2]
        if (max(abs(step)) < tol) break
      }
      alpha[j] <- a; beta[j] <- b
    }
    for (i in seq_len(n)) {
      yi <- M[i, ]; mask <- !is.na(yi)
      aj <- alpha[mask]; bj <- beta[mask]; yim <- yi[mask]
      xi <- x_hat[i]
      for (k in 1:5) {
        z <- aj * (xi - bj); p <- logistic(z); w <- p * (1 - p) + 1e-9
        g <- sum(aj * (yim - p))
        H <- -sum(w * aj^2)
        if (abs(H) < 1e-12) break
        step <- g / H
        xi <- xi - step
        if (abs(step) < tol) break
      }
      x_hat[i] <- xi
    }
    x_hat <- (x_hat - mean(x_hat)) / (stats::sd(x_hat) + 1e-12)
    Z <- outer(x_hat, beta, FUN = function(x, b) x - b)
    Z <- sweep(Z, 2L, alpha, "*")
    P <- logistic(Z)
    mask_full <- !is.na(M)
    ll <- sum(ifelse(mask_full,
                     M * log(P + 1e-12) + (1 - M) * log(1 - P + 1e-12),
                     0))
    if (abs(ll - prev_ll) < tol * max(1, abs(prev_ll))) break
    prev_ll <- ll
  }
  list(x_hat = x_hat, alpha = alpha, beta = beta,
       loglik = ll, n_iter = it, method = "irt_spatial_2pl")
}

#' @keywords internal
#' @rdname irtsp
#' @export
irt_spatial <- irtsp
