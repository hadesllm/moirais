# SPDX-License-Identifier: MIT OR Apache-2.0
#' Spatial GLM (Gaussian-identity case via profile ML).
#'
#' Y = X beta + delta + eps, delta ~ GP(0, sigma2 R_phi), R_phi exponential.
#' Profile-likelihood ML over phi; beta via GLS.
#'
#' @param x Numeric design matrix (n by p).
#' @param y Numeric response, length n.
#' @param coords Coordinate matrix (n by d).
#' @param family Character. Only "gaussian" supported in v0.2.
#' @return Named list: estimate, se, sigma2, phi, tau2, n, method.
#' @references Schabenberger & Gotway (2005), Ch 5.
#' @export
sglm <- function(x, y, coords, family = "gaussian") {
  if (family != "gaussian")
    stop("sglm: family=", family, " needs PQL/Laplace; tracker for v0.3.0")
  X <- as.matrix(x); y <- as.numeric(y); n <- length(y)
  coords <- if (is.matrix(coords)) coords else
    matrix(as.numeric(unlist(coords)), nrow = n)
  p <- ncol(X)
  if (nrow(X) != n || nrow(coords) != n)
    stop("shape mismatch among x, y, coords")
  D <- as.matrix(stats::dist(coords))
  h_max <- max(D)
  neg_ll <- function(log_phi) {
    phi <- exp(log_phi)
    R <- exp(-D / phi) + 1e-8 * diag(n)
    L <- tryCatch(chol(R), error = function(e) NULL)
    if (is.null(L)) return(1e12)
    Xw <- backsolve(L, X, transpose = TRUE)
    yw <- backsolve(L, y, transpose = TRUE)
    beta <- solve(crossprod(Xw), crossprod(Xw, yw))
    resid <- yw - Xw %*% beta
    sigma2 <- sum(resid ^ 2) / n
    logdet_R <- 2 * sum(log(diag(L)))
    0.5 * (n * log(2 * pi * sigma2) + logdet_R + n)
  }
  res <- stats::optimize(neg_ll, interval = c(log(max(h_max / 100, 1e-3)),
                                              log(max(h_max * 3, 1))))
  phi <- exp(res$minimum)
  R <- exp(-D / phi) + 1e-8 * diag(n)
  L <- chol(R)
  Xw <- backsolve(L, X, transpose = TRUE)
  yw <- backsolve(L, y, transpose = TRUE)
  XtSiX <- crossprod(Xw)
  beta <- as.numeric(solve(XtSiX, crossprod(Xw, yw)))
  resid <- yw - Xw %*% beta
  sigma2 <- as.numeric(sum(resid ^ 2) / max(n - p, 1))
  se_beta <- sqrt(diag(sigma2 * solve(XtSiX)))
  list(estimate = beta, se = se_beta, sigma2 = sigma2, phi = phi,
       tau2 = 0, n = n,
       method = "Spatial GLM (Gaussian, exponential covariance, ML)")
}

# CANONICAL TEST
# sglm(cbind(1, 0:4), 1 + 2*(0:4), matrix(0:4, ncol=1))$estimate # ~ c(1,2)

#' @rdname sglm
#' @keywords internal
#' @export
spatial_glm <- sglm
