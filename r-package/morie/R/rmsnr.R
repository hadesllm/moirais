# SPDX-License-Identifier: AGPL-3.0-or-later

#' Root-mean-square normalisation (Zhang & Sennrich 2019)
#'
#' @param x Input matrix (rows = tokens, cols = features).
#' @param gamma Optional learnable scale vector.
#' @param eps Numeric numerical-stability epsilon (default 1e-6).
#' @return Named list with tensor, rms, eps, method.
#' @keywords internal
rms_norm <- function(x, gamma = NULL, eps = 1e-6) {
  xm <- as.matrix(x)
  rms <- sqrt(rowMeans(xm * xm) + eps)
  y <- sweep(xm, 1L, rms, "/")
  if (!is.null(gamma)) y <- sweep(y, 2L, as.numeric(gamma), "*")
  list(tensor = y, rms = rms, eps = eps, method = "RMSNorm")
}
