# SPDX-License-Identifier: AGPL-3.0-or-later

#' Series-Tikhonov nonparametric instrumental variables
#'
#' @param x Numeric endogenous covariate vector.
#' @param y Numeric response vector.
#' @param z Numeric instrument vector.
#' @param J Integer basis size (default 5).
#' @param alpha Tikhonov regularisation parameter (default 1e-3).
#' @param grid Optional evaluation grid.
#' @param .bootstrap Logical; whether to bootstrap SEs (default TRUE).
#' @return Named list with estimate, se, grid, J, alpha, n, method.
#' @keywords internal
hrzn1 <- function(x, y, z, J = 5, alpha = 1e-3, grid = NULL,
                  .bootstrap = TRUE) {
  x <- as.numeric(x); y <- as.numeric(y); z <- as.numeric(z)
  n <- length(y)
  if (n < 50 || length(x) != n || length(z) != n) {
    # 2SLS fallback
    Xc <- cbind(1, x); Zc <- cbind(1, z)
    Pz <- Zc %*% MASS::ginv(t(Zc) %*% Zc) %*% t(Zc)
    beta <- MASS::ginv(t(Xc) %*% Pz %*% Xc) %*% (t(Xc) %*% Pz %*% y)
    return(list(estimate = as.numeric(beta[2]), se = NA_real_, n = n,
                method = "NPIV fallback: linear 2SLS"))
  }
  x_s <- (x - mean(x)) / max(stats::sd(x), 1e-6)
  z_s <- (z - mean(z)) / max(stats::sd(z), 1e-6)
  Bx <- .hrz_hermite(x_s, J); Bz <- .hrz_hermite(z_s, J)
  M <- (t(Bz) %*% Bx) / n
  BzY <- as.numeric((t(Bz) %*% y) / n)
  BzBz <- (t(Bz) %*% Bz) / n
  inv_BzBz <- MASS::ginv(BzBz + alpha * diag(J))
  A <- t(M) %*% inv_BzBz %*% M + alpha * diag(J)
  rhs <- t(M) %*% inv_BzBz %*% BzY
  coef <- solve(A, rhs)
  if (is.null(grid)) grid <- seq(min(x), max(x), length.out = 21)
  grid <- as.numeric(grid)
  grid_s <- (grid - mean(x)) / max(stats::sd(x), 1e-6)
  Bx_g <- .hrz_hermite(grid_s, J)
  g_hat <- as.numeric(Bx_g %*% coef)
  # Bootstrap SE (guarded against recursion explosion)
  if (.bootstrap) {
    set.seed(0); B <- 30
    boot <- matrix(0, B, length(grid))
    for (b in 1:B) {
      idx <- sample.int(n, n, replace = TRUE)
      sub <- tryCatch(hrzn1(x[idx], y[idx], z[idx], J = J, alpha = alpha,
                            grid = grid, .bootstrap = FALSE),
                      error = function(e) list(estimate = g_hat))
      boot[b, ] <- as.numeric(sub$estimate)
    }
    se <- apply(boot, 2, stats::sd)
  } else {
    se <- rep(NA_real_, length(grid))
  }
  list(estimate = g_hat, se = as.numeric(se), grid = grid, J = J, alpha = alpha,
       n = n, method = "Series-Tikhonov NPIV on Hermite basis")
}

# canonical full-name alias (Py<->R API parity)
#' @rdname hrzn1
#' @keywords internal
#' @export
horowitz_nonparametric_iv <- hrzn1
