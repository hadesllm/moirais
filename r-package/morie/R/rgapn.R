#' Approximate entropy -- Rangayyan Ch 7
#'
#' Pincus (1991) approximate entropy.
#'
#' \deqn{\mathrm{ApEn}(m, r) = \phi_m(r) - \phi_{m+1}(r)}
#'
#' Self-matches INCLUDED (Pincus convention) and Chebyshev distance.
#'
#' @param x Numeric vector.
#' @param m Template length (default 2).
#' @param r Tolerance (default `0.2 * sd(x)`).
#' @return Named list `ApEn`, `phi_m`, `phi_m1`, `m`, `r`, `n`.
#' @references Pincus (1991), PNAS 88:2297.
#' @export
#' @examples
#' set.seed(0); rgapn(rnorm(100), m = 2)$ApEn
rgapn <- function(x, m = 2L, r = NULL) {
  N <- length(x)
  if (is.null(r)) r <- 0.2 * stats::sd(x)
  m <- as.integer(m)
  if (N <= m + 1) stop("Need length(x) > m + 1.")
  phi <- function(mm) {
    nT <- N - mm + 1
    M <- matrix(0, nrow = nT, ncol = mm)
    for (i in seq_len(nT)) M[i, ] <- x[i:(i + mm - 1)]
    C <- numeric(nT)
    for (i in seq_len(nT)) {
      d <- apply(abs(sweep(M, 2, M[i, ])), 1, max)
      C[i] <- sum(d <= r) / nT
    }
    C <- pmax(C, 1e-30)
    mean(log(C))
  }
  pm <- phi(m); pm1 <- phi(m + 1L)
  list(ApEn = pm - pm1, phi_m = pm, phi_m1 = pm1, m = m, r = r, n = N)
}

#' @rdname rgapn
#' @keywords internal
#' @export
rangayyan_approximate_entropy <- rgapn
