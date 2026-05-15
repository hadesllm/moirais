# SPDX-License-Identifier: AGPL-3.0-or-later

#' GBLUP -- full mixed-model implementation
#'
#' Solves Henderson's MME with VanRaden G.
#'
#' @param x Fixed-effect design (vector or matrix).
#' @param y Numeric response.
#' @param markers Genotype matrix (n x m).
#' @param lambda_gblup Optional ratio sigma_e^2 / sigma_g^2 (default h^2=0.5).
#' @return Named list (estimate, g_hat, beta, se, lambda_gblup, n, method).
#' @references Montesinos Lopez Ch 3.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
gblup_full <- function(x, y, markers, lambda_gblup = NULL) {
  y <- as.numeric(y); n <- length(y)
  M <- as.matrix(markers)
  G <- grm_vanraden(M, method = 1)$estimate
  G <- G + 1e-6 * diag(n)
  cand <- if (is.null(x) || (is.numeric(x) && length(x) == 0)) {
    matrix(1, n, 1)
  } else {
    cbind(1, as.matrix(x))
  }
  qrx <- qr(cand)
  X <- cand[, qrx$pivot[seq_len(qrx$rank)], drop = FALSE]
  if (is.null(lambda_gblup)) {
    var_y <- if (n > 1) stats::var(y) else 1
    lam <- (0.5 * var_y) / (0.5 * var_y)  # = 1 with h^2=0.5
  } else {
    lam <- as.numeric(lambda_gblup)
  }
  Ginv <- solve(G)
  p <- ncol(X)
  C <- rbind(cbind(crossprod(X), t(X)),
             cbind(X,            diag(n) + lam * Ginv))
  rhs <- c(crossprod(X, y), y)
  sol <- solve(C, rhs)
  beta <- sol[seq_len(p)]
  g_hat <- sol[(p + 1):(p + n)]
  y_hat <- X %*% beta + g_hat
  resid <- y - as.numeric(y_hat)
  se <- sqrt(sum(resid^2) / max(n - p, 1))
  list(estimate = mean(g_hat), g_hat = g_hat, beta = beta,
       se = se, y_hat = as.numeric(y_hat),
       lambda_gblup = lam, n = n,
       method = "GBLUP with VanRaden G")
}

# CANONICAL TEST
# set.seed(0); M <- matrix(sample(0:2, 20, TRUE), 4, 5)
# gblup_full(rep(0, 4), c(1, 2, 3, 2.5), M)
