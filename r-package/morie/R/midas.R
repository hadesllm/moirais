# SPDX-License-Identifier: MIT OR Apache-2.0

#' MIDAS regression with Beta-polynomial weights
#'
#' @param x High-frequency regressor matrix (n_t x K) or flat vector.
#' @param y Low-frequency target (length n_t).
#' @param K Number of high-frequency lags (required when x is flat).
#' @return Named list with \code{beta0, beta1, theta1, theta2, weights,
#'   r2, n, K, method}.
#' @export
midas_regression <- function(x, y, K = NULL) {
  Y <- as.numeric(y); nT <- length(Y)
  if (is.null(dim(x))) {
    if (is.null(K)) stop("Pass K when x is flat.")
    if (length(x) < K + nT - 1) stop("x too short.")
    Xf <- as.numeric(x)
    rows <- vector("list", nT)
    for (t in seq_len(nT)) {
      end <- length(Xf) - (nT - t)
      rows[[t]] <- rev(Xf[(end - K + 1):end])
    }
    X <- do.call(rbind, rows)
  } else {
    X <- as.matrix(x); K <- ncol(X)
  }
  if (nrow(X) != nT) stop("Dim mismatch.")
  if (nT < 4) stop("Need >=4 obs.")
  neg_ll <- function(p) {
    b0 <- p[1]; b1 <- p[2]; t1 <- p[3]; t2 <- p[4]
    if (t1 <= 0 || t2 <= 0) return(1e10)
    w <- .morie_beta_weights(t1, t2, K)
    yhat <- b0 + b1 * (X %*% w)
    sse <- sum((Y - yhat)^2)
    if (!is.finite(sse)) 1e10 else sse
  }
  opt <- nlminb(c(mean(Y), 1, 1.5, 2), neg_ll,
                lower = c(-1e3, -1e3, 0.1, 0.1),
                upper = c( 1e3,  1e3, 50,  50))
  b0 <- opt$par[1]; b1 <- opt$par[2]
  t1 <- opt$par[3]; t2 <- opt$par[4]
  w <- .morie_beta_weights(t1, t2, K)
  resid <- Y - (b0 + b1 * (X %*% w))
  ss_tot <- sum((Y - mean(Y))^2)
  r2 <- if (ss_tot > 0) 1 - sum(resid^2) / ss_tot else NA_real_
  list(beta0 = b0, beta1 = b1, theta1 = t1, theta2 = t2,
       weights = w, r2 = r2, n = nT, K = K,
       method = "MIDAS Beta-polynomial via nlminb (base R)")
}
