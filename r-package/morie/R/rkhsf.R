# SPDX-License-Identifier: MIT OR Apache-2.0

#' RKHS regression with Gaussian kernel
#'
#' @param x Fixed-effect design.
#' @param y Numeric response.
#' @param markers Genotype matrix (n x m).
#' @param h Kernel bandwidth (default median ||m_i - m_j||^2).
#' @param lam Ridge regulariser on alpha (default 1).
#' @return list(estimate, alpha, beta, K, f_hat, se, h, n, method).
#' @references Gianola & van Kaam (2008). Montesinos Lopez Ch 5.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
rkhs_full <- function(x, y, markers, h = NULL, lam = 1) {
  y <- as.numeric(y); n <- length(y)
  M <- as.matrix(markers)
  sq <- rowSums(M^2)
  D2 <- pmax(outer(sq, sq, "+") - 2 * tcrossprod(M), 0)
  if (is.null(h)) {
    iu <- D2[upper.tri(D2)]
    h <- if (length(iu) > 0) stats::median(iu) else 1
    if (!is.finite(h) || h <= 0) h <- 1
  }
  K <- exp(-D2 / h)
  cand <- if (is.null(x) || (is.numeric(x) && length(x) == 0)) {
    matrix(1, n, 1)
  } else cbind(1, as.matrix(x))
  qrx <- qr(cand); X <- cand[, qrx$pivot[seq_len(qrx$rank)], drop = FALSE]
  beta <- stats::lsfit(X, y, intercept = FALSE)$coefficients
  r <- y - as.numeric(X %*% beta)
  alpha <- solve(K + lam * diag(n), r)
  f_hat <- as.numeric(K %*% alpha)
  y_hat <- as.numeric(X %*% beta) + f_hat
  resid <- y - y_hat
  se <- sqrt(sum(resid^2) / max(n - ncol(X), 1))
  list(estimate = mean(f_hat), alpha = alpha, beta = beta, K = K,
       f_hat = f_hat, se = se, h = h, n = n,
       method = "RKHS regression (Gaussian kernel)")
}

# CANONICAL TEST
# set.seed(1); M <- matrix(sample(0:2, 20, TRUE), 5, 4)
# rkhs_full(rep(0,5), c(1,2,1.5,2.5,2), M)
