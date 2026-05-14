# SPDX-License-Identifier: MIT OR Apache-2.0
#' Sobol quasi-random sequence (Sobol 1967)
#'
#' Thin wrapper over \code{randtoolbox::sobol} (low-discrepancy in the unit cube of dimension d).
#' Falls back to a pure-R Halton sequence if randtoolbox is unavailable.
#'
#' @param N integer; default 128.
#' @param d integer; default 1.
#' @param f optional integrand; returns scalar.
#' @param scramble logical; Owen scrambling (default TRUE).
#' @param seed integer.
#' @return list: sample, estimate (if f given), se, N, d, method.
#' @importFrom utils getFromNamespace
#' @keywords internal
sobls <- function(N = 128L, d = 1L, f = NULL, scramble = TRUE, seed = 42L) {
  sample <- NULL
  if (requireNamespace("randtoolbox", quietly = TRUE)) {
    sobol_fn <- getFromNamespace("sobol", "randtoolbox")
    sample <- sobol_fn(n = as.integer(N), dim = as.integer(d),
                       scrambling = if (scramble) 1L else 0L, seed = seed)
    if (!is.matrix(sample)) sample <- matrix(sample, ncol = d)
  } else {
    # Halton sequence fallback (pure R)
    primes <- c(2, 3, 5, 7, 11, 13, 17, 19, 23, 29)[seq_len(d)]
    halton <- function(i, b) {
      f <- 1; r <- 0
      while (i > 0) {
        f <- f / b
        r <- r + f * (i %% b)
        i <- floor(i / b)
      }
      r
    }
    sample <- matrix(0, N, d)
    for (j in seq_len(d))
      sample[, j] <- vapply(seq_len(N), halton, numeric(1), b = primes[j])
  }
  out <- list(sample = sample, N = as.integer(N), d = as.integer(d),
              method = "Sobol QMC (Sobol 1967)")
  if (!is.null(f)) {
    fv <- apply(sample, 1, f)
    out$estimate <- mean(fv)
    out$se <- stats::sd(fv) / sqrt(N)
  }
  out
}

# CANONICAL TEST
# r <- sobls(N = 128, d = 2, f = function(u) u[1] * u[2], seed = 0)
# stopifnot(abs(r$estimate - 0.25) < 0.05)

#' @rdname sobls
#' @keywords internal
#' @export
sobol_sequence <- sobls
