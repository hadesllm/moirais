# SPDX-License-Identifier: AGPL-3.0-or-later

#' Discrete wavelet decomposition for a time series
#'
#' @param x Numeric univariate series.
#' @param wavelet Wavelet family. Default "haar".
#' @param level Decomposition depth. Default floor(log2 n) capped at 6.
#' @return Named list with \code{approximation, details, energies, level,
#'   n, wavelet, method}.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
wavelet_time_series <- function(x, wavelet = "haar", level = NULL) {
  y <- as.numeric(x); n <- length(y)
  if (n < 4) stop("Need >=4 obs.")
  max_lv <- floor(log2(n))
  if (is.null(level)) level <- min(max(max_lv, 1), 6)
  level <- min(level, max_lv)
  if (requireNamespace("wavelets", quietly = TRUE)) {
    fit <- wavelets::dwt(y, filter = wavelet, n.levels = level)
    cA <- as.numeric(fit@V[[level]])
    cDs <- lapply(rev(fit@W), as.numeric)
    energies <- c(sum(cA^2), sapply(cDs, function(c) sum(c^2)))
    return(list(approximation = cA,
                details = cDs,
                energies = energies,
                level = level, n = n, wavelet = wavelet,
                method = sprintf("DWT via wavelets (wavelet=%s, level=%d)",
                                 wavelet, level)))
  }
  cA <- y; cDs <- list()
  for (lv in seq_len(level)) {
    if (length(cA) < 2) break
    if (length(cA) %% 2 == 1) cA <- c(cA, cA[length(cA)])
    even <- cA[seq(1, length(cA), 2)]
    odd <- cA[seq(2, length(cA), 2)]
    cD <- (even - odd) / sqrt(2)
    cA <- (even + odd) / sqrt(2)
    cDs <- c(list(cD), cDs)
  }
  energies <- c(sum(cA^2), sapply(cDs, function(c) sum(c^2)))
  list(approximation = cA, details = cDs, energies = energies,
       level = level, n = n, wavelet = "haar",
       method = "Haar DWT (base R fallback)")
}
