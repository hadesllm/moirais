# SPDX-License-Identifier: MIT OR Apache-2.0
#' Spatial autoregressive error model (SAR error, ML).
#'
#' Y = X beta + u,  u = lambda W u + eps,  eps ~ N(0, sigma2 I).
#' Concentrated log-likelihood in lambda; beta via GLS on the
#' transformed system A y = A X beta + eps, A = I - lambda W.
#'
#' @param x Design matrix (n by p, intercept explicit).
#' @param y Response, length n.
#' @param w Row-standardised n-by-n weights matrix.
#' @return Named list: estimate, se, lambda, sigma2, n, method.
#' @references Anselin (1988); Schabenberger & Gotway (2005), Ch 7.
#' @export
sarre <- function(x, y, w) {
  X <- as.matrix(x); y <- as.numeric(y); W <- as.matrix(w)
  n <- length(y); p <- ncol(X)
  if (nrow(X) != n || any(dim(W) != c(n, n)))
    stop("shape mismatch among x, y, w")
  I <- diag(n)
  neg_ll <- function(lam) {
    A <- I - lam * W
    AX <- A %*% X; Ay <- A %*% y
    beta <- tryCatch(solve(crossprod(AX), crossprod(AX, Ay)),
                     error = function(e) NULL)
    if (is.null(beta)) return(1e12)
    e <- Ay - AX %*% beta
    sigma2 <- as.numeric(sum(e ^ 2)) / n
    if (sigma2 <= 0) return(1e12)
    det_sign <- determinant(A, logarithm = TRUE)
    if (det_sign$sign <= 0) return(1e12)
    logdetA <- as.numeric(det_sign$modulus)
    0.5 * n * log(2 * pi * sigma2) - logdetA + 0.5 * n
  }
  res <- stats::optimize(neg_ll, interval = c(-0.99, 0.99))
  lam <- res$minimum
  A <- I - lam * W
  AX <- A %*% X; Ay <- A %*% y
  beta <- as.numeric(solve(crossprod(AX), crossprod(AX, Ay)))
  e <- as.numeric(Ay - AX %*% beta)
  sigma2 <- as.numeric(sum(e ^ 2)) / max(n - p, 1)
  cov_b <- sigma2 * solve(crossprod(AX))
  se <- sqrt(pmax(diag(cov_b), 0))
  list(estimate = beta, se = se, lambda = lam, sigma2 = sigma2,
       n = n, method = "SAR error (ML, concentrated log-likelihood)")
}

# CANONICAL TEST  (with row-standardised path graph)

#' @rdname sarre
#' @keywords internal
#' @export
spatial_ar_error <- sarre
