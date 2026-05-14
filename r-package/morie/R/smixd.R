# SPDX-License-Identifier: MIT OR Apache-2.0
#' Spatial linear mixed model via REML.
#'
#' Y = X beta + delta + eps,
#'   delta ~ N(0, sigma2 R_phi),  R_phi exponential,
#'   eps ~ N(0, tau2 I).
#'
#' @param x Numeric design matrix (n by p).
#' @param y Numeric response.
#' @param coords Coordinate matrix.
#' @return Named list: estimate, se, sigma2, tau2, phi, n, method.
#' @references Patterson & Thompson (1971); Schabenberger & Gotway (2005), Ch 5.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
smixd <- function(x, y, coords) {
  X <- as.matrix(x); y <- as.numeric(y); n <- length(y)
  coords <- if (is.matrix(coords)) coords else
    matrix(as.numeric(unlist(coords)), nrow = n)
  p <- ncol(X)
  D <- as.matrix(stats::dist(coords))
  h_max <- max(D, 1)
  neg_reml <- function(theta) {
    phi <- exp(theta[1]); nu <- exp(theta[2])
    Sigma <- exp(-D / phi) + nu * diag(n)
    L <- tryCatch(chol(Sigma), error = function(e) NULL)
    if (is.null(L)) return(1e12)
    Xw <- backsolve(L, X, transpose = TRUE)
    yw <- backsolve(L, y, transpose = TRUE)
    XtSiX <- crossprod(Xw)
    L2 <- tryCatch(chol(XtSiX), error = function(e) NULL)
    if (is.null(L2)) return(1e12)
    beta <- as.numeric(solve(XtSiX, crossprod(Xw, yw)))
    resid <- yw - Xw %*% beta
    sigma2 <- as.numeric(sum(resid ^ 2)) / max(n - p, 1)
    if (sigma2 <= 0) return(1e12)
    logdet_S <- 2 * sum(log(diag(L)))
    logdet_K <- 2 * sum(log(diag(L2)))
    0.5 * (logdet_S + logdet_K + (n - p) * log(2 * pi * sigma2) + (n - p))
  }
  res <- stats::optim(c(log(h_max / 3), log(0.1)), neg_reml,
                      method = "Nelder-Mead",
                      control = list(maxit = 400, reltol = 1e-6))
  phi <- exp(res$par[1]); nu <- exp(res$par[2])
  Sigma <- exp(-D / phi) + nu * diag(n)
  L <- chol(Sigma)
  Xw <- backsolve(L, X, transpose = TRUE)
  yw <- backsolve(L, y, transpose = TRUE)
  XtSiX <- crossprod(Xw)
  beta <- as.numeric(solve(XtSiX, crossprod(Xw, yw)))
  resid <- yw - Xw %*% beta
  sigma2 <- as.numeric(sum(resid ^ 2)) / max(n - p, 1)
  tau2 <- nu * sigma2
  se_beta <- sqrt(diag(sigma2 * solve(XtSiX)))
  list(estimate = beta, se = se_beta, sigma2 = sigma2, tau2 = tau2,
       phi = phi, n = n,
       method = "Spatial linear mixed model (REML, exponential covariance)")
}

# CANONICAL TEST
# smixd(cbind(1, 0:4), 1 + 2*(0:4), matrix(0:4, ncol=1))$estimate # ~ c(1,2)

#' @rdname smixd
#' @keywords internal
#' @export
spatial_mixed_model <- smixd
