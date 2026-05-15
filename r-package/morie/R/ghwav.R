# SPDX-License-Identifier: AGPL-3.0-or-later

#' Haar-wavelet spike-and-slab BayesThresh estimator (Abramovich 1998).
#'
#' @param x Numeric data vector.
#' @param pi Numeric prior inclusion probability (default 0.5).
#' @param sigma Optional slab sd.
#' @param noise Optional noise sd.
#' @return Named list with estimate, fitted, noise, sigma, inclusion, n, method.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
ghosal_wavelet_prior <- function(x, pi = 0.5, sigma = NULL, noise = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (n < 4) return(list(estimate = if (n) mean(x) else NA_real_,
                          fitted = x, n = n, method = "Wavelet prior (n<4)"))
  dw <- .gh_haar_dwt(x); coeffs <- dw$coeffs; L <- dw$L
  finest <- coeffs[[1]]
  if (is.null(noise)) noise <- max(stats::mad(finest) / 0.6745, 1e-6)
  if (is.null(sigma)) {
    all_d <- unlist(coeffs[-length(coeffs)])
    sigma <- sqrt(max(var(all_d) - noise^2, 1e-6))
  }
  sigma <- max(sigma, 1e-6)
  incl <- c(); new_coeffs <- list()
  for (i in seq_along(coeffs[-length(coeffs)])) {
    d <- coeffs[[i]]
    var_slab <- sigma^2 + noise^2
    log_slab  <- stats::dnorm(d, 0, sqrt(var_slab), log = TRUE)
    log_spike <- stats::dnorm(d, 0, noise, log = TRUE)
    a <- log(pi) + log_slab; b <- log(1 - pi) + log_spike
    mm <- pmax(a, b)
    w <- exp(a - mm) / (exp(a - mm) + exp(b - mm))
    shrink <- sigma^2 / var_slab
    new_coeffs[[i]] <- w * shrink * d
    incl <- c(incl, w)
  }
  new_coeffs[[length(new_coeffs) + 1L]] <- coeffs[[length(coeffs)]]
  # Inverse DWT
  cur <- new_coeffs[[length(new_coeffs)]]
  for (i in (length(new_coeffs) - 1L):1L) {
    d <- new_coeffs[[i]]
    out <- numeric(2 * length(cur))
    out[seq(1, length(out), by = 2)] <- (cur + d) / sqrt(2)
    out[seq(2, length(out), by = 2)] <- (cur - d) / sqrt(2)
    cur <- out
  }
  fitted <- cur[seq_len(n)]
  list(estimate = mean(fitted), fitted = fitted, noise = noise,
       sigma = sigma, inclusion = mean(incl), n = n,
       method = "Haar-wavelet spike-and-slab BayesThresh")
}
