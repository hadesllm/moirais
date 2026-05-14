# SPDX-License-Identifier: MIT OR Apache-2.0
#' Geographically weighted regression (GWR).
#'
#' Local WLS at each site i:
#' \deqn{\beta(s_i) = (X^\top W(s_i) X)^{-1} X^\top W(s_i) y}.
#'
#' @param x Design matrix (n by k).
#' @param y Response, length n.
#' @param coords Coordinate matrix.
#' @param bandwidth Kernel bandwidth; default = median pairwise distance.
#' @param kernel "gaussian" or "bisquare".
#' @return Named list: estimate (n by k list), se, bandwidth, kernel,
#'   n, method.
#' @references Brunsdon, Fotheringham & Charlton (1996).
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
gwreg <- function(x, y, coords, bandwidth = NULL, kernel = "gaussian") {
  X <- as.matrix(x); y <- as.numeric(y); n <- length(y)
  coords <- if (is.matrix(coords)) coords else
    matrix(as.numeric(unlist(coords)), nrow = n)
  k <- ncol(X)
  D <- as.matrix(stats::dist(coords))
  if (is.null(bandwidth)) bandwidth <- stats::median(D[D > 0])
  betas <- matrix(0, n, k); ses <- matrix(0, n, k)
  for (i in seq_len(n)) {
    d <- D[i, ]
    w <- if (kernel == "bisquare")
      ifelse(d <= bandwidth, (1 - (d / bandwidth) ^ 2) ^ 2, 0)
    else exp(-0.5 * (d / bandwidth) ^ 2)
    sw <- sqrt(w)
    Xw <- X * sw; yw <- y * sw
    XtWX <- crossprod(Xw)
    bi <- tryCatch(solve(XtWX, crossprod(Xw, yw)),
                   error = function(e) qr.solve(XtWX, crossprod(Xw, yw)))
    betas[i, ] <- as.numeric(bi)
    resid <- yw - Xw %*% bi
    df <- max(sum(w) - k, 1)
    sigma2 <- as.numeric(sum(resid ^ 2)) / df
    cov_i <- tryCatch(sigma2 * solve(XtWX),
                      error = function(e) matrix(NA_real_, k, k))
    ses[i, ] <- sqrt(pmax(diag(cov_i), 0))
  }
  list(estimate = betas, se = ses, bandwidth = bandwidth, kernel = kernel,
       n = n, method = sprintf("GWR (%s kernel)", kernel))
}

# CANONICAL TEST
# gwreg(cbind(1, 0:4), 0:4, matrix(0:4, ncol=1))$estimate
# row-wise -> approx c(0, 1) at every site

#' @rdname gwreg
#' @keywords internal
#' @export
geographically_weighted_regression <- gwreg
