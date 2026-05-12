# SPDX-License-Identifier: GPL-2.0-only

#' Kernel density estimator (Rosenblatt-Parzen)
#'
#' @param x Numeric evaluation points (or sample if \code{sample} is NULL).
#' @param bandwidth Optional kernel bandwidth (Silverman default).
#' @param sample Optional numeric sample to estimate from.
#' @return Named list with estimate, se, bandwidth, n, kernel, method.
#' @keywords internal
hrzk1 <- function(x, bandwidth = NULL, sample = NULL) {
  if (is.null(sample)) {
    data <- as.numeric(x); grid <- data
  } else {
    data <- as.numeric(sample); grid <- as.numeric(x)
  }
  n <- length(data)
  if (n < 2) return(list(estimate = NA_real_, se = NA_real_, n = n,
                          method = "kernel-density (insufficient data)"))
  h <- if (is.null(bandwidth)) .hrz_silverman(data) else as.numeric(bandwidth)
  if (h <= 0) h <- .hrz_silverman(data)
  diffs <- outer(grid, data, `-`) / h
  w <- exp(-0.5 * diffs^2) / sqrt(2 * pi)
  f_hat <- rowMeans(w) / h
  se <- sqrt(pmax(f_hat, 0) * .hrz_R_K_gaussian / (n * h))
  list(estimate = if (length(f_hat) == 1) as.numeric(f_hat) else f_hat,
       se = if (length(se) == 1) as.numeric(se) else se,
       bandwidth = h, n = n, kernel = "gaussian",
       method = "Rosenblatt-Parzen kernel density")
}

# canonical full-name alias (Py<->R API parity)
#' @rdname hrzk1
#' @keywords internal
#' @export
horowitz_kernel_density <- hrzk1
